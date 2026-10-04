// gather_prototype.cu — THROWAWAY research prototype (NOT product code).
//
// Implements the `logprob_topk` Op described in
// docs/research/logprobs-evidence.md §2.3, plus a driver that materializes inputs
// and kernel outputs as raw little-endian files (--gen) and a CUDA-event
// benchmark (--bench).
//
// Formula (§2.2), per (vocab v, row r):
//   T          = temperature[r] > 0 ? temperature[r] : 1
//   adjusted_v = decoded_bf16(logits[v,r]) - presence[r]*(counts[r,v] > 0)
//                                         - frequency[r]*counts[r,v]
//   scaled_v   = adjusted_v / T
//   lse_r      = logsumexp_v scaled_v            (max-subtracted, -inf safe)
//   logprob_v  = scaled_v - lse_r
//
// Outputs: top_ids I32 [20, rows] and top_values FP32 [20, rows] (scaled
// descending, LOWER token id breaking ties), lse FP32 [rows], chosen I32 [rows]
// (the top-1 id; the prototype does not sample).
//
// Structure (§2.3 says "one CTA per row"; the prototype splits each row across
// SPLIT CTAs so the vocabulary reduction uses the whole GPU — a single CTA of
// 256 threads per row is memory-latency bound and measured ~340 us/row, ~40x a
// plain vocabulary read). Two kernels:
//   partials: grid (SPLIT, rows) — per-chunk [begin,end) max, sum(exp) and
//             bounded top-K (dynamic-index insertion + shared tree merge).
//   combine:  grid (rows) — row max over partials, sum(exp) combine, and a
//             merge of the SPLIT top-K lists into the final top-20.
// A speculative block is reproduced by passing rows = batch*(k+1).
//
// File layouts (all little-endian):
//   logits.bf16   uint16 [V, rows]   row-major, index v*rows + r
//   counts.i32    int32  [rows, V]   row-major, index r*V + v
//   top_ids.i32   int32  [20, rows]  index k*rows + r
//   top_values.f32 float [20, rows]  index k*rows + r
//   lse.f32       float  [rows]
//   chosen.i32    int32  [rows]
//
// CLI:
//   gather_prototype --gen   <case>[:rN[wM]] <outdir>
//   gather_prototype --bench <case>[:rN[wM]]
// where <case> is one of: plain-t1 plain-t07 greedy penalty masked mask-penalty spec-k4
// and the optional suffix sets rows (default 1) and speculative width (default 1).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x)                                                                      \
  do {                                                                             \
    cudaError_t e_ = (x);                                                          \
    if (e_ != cudaSuccess) {                                                        \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(1);                                                                      \
    }                                                                              \
  } while (0)

constexpr int K = 20;         // top-K (OpenAI maximum)
constexpr int THREADS = 256;  // threads per CTA
constexpr int SPLIT = 128;    // CTAs per row for the partial pass
constexpr int V = 151936;     // Qwen3.5 vocab

// ---------------------------------------------------------------------------
// Device side
// ---------------------------------------------------------------------------

struct Cand {
  float v;
  int id;
};

__device__ __forceinline__ bool better(float av, int aid, float bv, int bid) {
  return (av > bv) || (av == bv && aid < bid);
}

__device__ __forceinline__ float block_max(float x, float* sRed, int t) {
  sRed[t] = x;
  __syncthreads();
  for (int st = THREADS / 2; st > 0; st >>= 1) {
    if (t < st) sRed[t] = fmaxf(sRed[t], sRed[t + st]);
    __syncthreads();
  }
  return sRed[0];
}

__device__ __forceinline__ float block_sum(float x, float* sRed, int t) {
  sRed[t] = x;
  __syncthreads();
  for (int st = THREADS / 2; st > 0; st >>= 1) {
    if (t < st) sRed[t] += sRed[t + st];
    __syncthreads();
  }
  return sRed[0];
}

// Tree-merge THREADS sorted candidate lists from shared into sCand[0..K).
__device__ __forceinline__ void block_merge_topk(Cand* sCand, int t) {
  for (int st = THREADS / 2; st > 0; st >>= 1) {
    if (t < st) {
      int i = 0, j = 0;
      Cand out[K];
#pragma unroll
      for (int k = 0; k < K; ++k) {
        bool take_a;
        if (i >= K)
          take_a = false;
        else if (j >= K)
          take_a = true;
        else
          take_a = !better(sCand[(t + st) * K + j].v, sCand[(t + st) * K + j].id,
                           sCand[t * K + i].v, sCand[t * K + i].id);
        out[k] = take_a ? sCand[t * K + i] : sCand[(t + st) * K + j];
        if (take_a) ++i; else ++j;
      }
#pragma unroll
      for (int k = 0; k < K; ++k) sCand[t * K + k] = out[k];
    }
    __syncthreads();
  }
}

// Per-CTA partial over the vocab chunk [begin,end) of row r.
__global__ void partials_kernel(const uint16_t* __restrict__ logits, // [V, rows]
                                const int32_t* __restrict__ counts,  // [rows, V]
                                const float* __restrict__ temperature,
                                const float* __restrict__ presence,
                                const float* __restrict__ frequency,
                                int V_, int rows, int split,
                                float* __restrict__ pmax,   // [rows*split]
                                float* __restrict__ psum,   // [rows*split]
                                float* __restrict__ ptop_v, // [rows*split*K]
                                int32_t* __restrict__ ptop_i) {
  __shared__ Cand sCand[THREADS * K];
  __shared__ float sRed[THREADS];

  const int s = blockIdx.x;
  const int r = blockIdx.y;
  const int t = threadIdx.x;
  const int chunk = (V_ + split - 1) / split;
  const int begin = s * chunk;
  const int end = min(begin + chunk, V_);
  const int pl = r * split + s;

  const float T = temperature[r] > 0.f ? temperature[r] : 1.f;
  const float invT = 1.f / T; // reciprocal multiply: IEEE division per element is costly
  const float pres = presence[r];
  const float freq = frequency[r];
  const long row_stride_l = rows;
  const long row_base_c = (long)r * V_;

  // ---- chunk max -------------------------------------------------------
  float lmax = -INFINITY;
  for (int v = begin + t; v < end; v += THREADS) {
    const float lf = __uint_as_float(((uint32_t)logits[(long)v * row_stride_l + r]) << 16);
    const int c = counts[row_base_c + v];
    const float adj = lf - pres * (c > 0 ? 1.f : 0.f) - freq * (float)c;
    lmax = fmaxf(lmax, adj * invT);
  }
  const float locmax = block_max(lmax, sRed, t);

  // ---- chunk sum(exp) + top-K ------------------------------------------
  Cand loc[K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    loc[k].v = -INFINITY;
    loc[k].id = 0x7fffffff;
  }
  float lsum = 0.f;
  if (locmax > -INFINITY) {
    for (int v = begin + t; v < end; v += THREADS) {
      const float lf = __uint_as_float(((uint32_t)logits[(long)v * row_stride_l + r]) << 16);
      const int c = counts[row_base_c + v];
      const float adj = lf - pres * (c > 0 ? 1.f : 0.f) - freq * (float)c;
      const float sv = adj * invT;
      lsum += expf(sv - locmax);
      if (better(sv, v, loc[K - 1].v, loc[K - 1].id)) {
        int p = K - 1;
        while (p > 0 && better(sv, v, loc[p - 1].v, loc[p - 1].id)) {
          loc[p] = loc[p - 1];
          --p;
        }
        loc[p].v = sv;
        loc[p].id = v;
      }
    }
  }
  const float psum_val = block_sum(lsum, sRed, t);

#pragma unroll
  for (int k = 0; k < K; ++k) sCand[t * K + k] = loc[k];
  __syncthreads();
  block_merge_topk(sCand, t);

  if (t == 0) {
    pmax[pl] = locmax;
    psum[pl] = psum_val;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      ptop_v[pl * K + k] = sCand[k].v;
      ptop_i[pl * K + k] = sCand[k].id;
    }
  }
}

// Combine the SPLIT partials of one row into the final logsumexp and top-20.
__global__ void combine_kernel(const float* __restrict__ pmax,
                               const float* __restrict__ psum,
                               const float* __restrict__ ptop_v,
                               const int32_t* __restrict__ ptop_i,
                               int rows, int split,
                               float* __restrict__ top_values, // [K, rows]
                               int32_t* __restrict__ top_ids,  // [K, rows]
                               float* __restrict__ lse,        // [rows]
                               int32_t* __restrict__ chosen,   // [rows]
                               int* __restrict__ invalid) {
  __shared__ Cand sCand[THREADS * K];
  __shared__ float sRed[THREADS];

  const int r = blockIdx.x;
  const int t = threadIdx.x;
  const int base = r * split;

  // ---- row max ---------------------------------------------------------
  float m = -INFINITY;
  for (int s = t; s < split; s += THREADS) m = fmaxf(m, pmax[base + s]);
  m = block_max(m, sRed, t);
  if (t == 0 && !(m > -INFINITY)) atomicExch(invalid, 1);

  // ---- logsumexp combine ----------------------------------------------
  float acc = 0.f;
  for (int s = t; s < split; s += THREADS) {
    const float pv = pmax[base + s];
    if (pv > -INFINITY) acc += psum[base + s] * expf(pv - m);
  }
  const float total = block_sum(acc, sRed, t);
  const float lse_val = m + logf(total);

  // ---- top-K merge over split*K candidates -----------------------------
  Cand loc[K];
#pragma unroll
  for (int k = 0; k < K; ++k) {
    loc[k].v = -INFINITY;
    loc[k].id = 0x7fffffff;
  }
  const int ncand = split * K;
  const long cand_base = (long)r * split * K;
  for (int i = t; i < ncand; i += THREADS) {
    const float cv = ptop_v[cand_base + i];
    const int ci = ptop_i[cand_base + i];
    if (better(cv, ci, loc[K - 1].v, loc[K - 1].id)) {
      int p = K - 1;
      while (p > 0 && better(cv, ci, loc[p - 1].v, loc[p - 1].id)) {
        loc[p] = loc[p - 1];
        --p;
      }
      loc[p].v = cv;
      loc[p].id = ci;
    }
  }
#pragma unroll
  for (int k = 0; k < K; ++k) sCand[t * K + k] = loc[k];
  __syncthreads();
  block_merge_topk(sCand, t);

  if (t == 0) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
      // sCand holds scaled logits; the reported value is the logprob.
      top_values[k * rows + r] = sCand[k].v - lse_val;
      top_ids[k * rows + r] = sCand[k].id;
    }
    lse[r] = lse_val;
    chosen[r] = sCand[0].id;
  }
}

// ---------------------------------------------------------------------------
// Host side: deterministic data generation
// ---------------------------------------------------------------------------

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  float u() { return (float)(next() >> 11) * (1.0f / 9007199254740992.0f); }
  float n() { // Box-Muller
    float u1 = u();
    if (u1 < 1e-12f) u1 = 1e-12f;
    float u2 = u();
    return sqrtf(-2.f * logf(u1)) * cosf(6.283185307179586f * u2);
  }
};

static uint16_t f32_to_bf16_rn(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t lsb = (x >> 16) & 1u;
  const uint32_t rounded = x + 0x7fffu + lsb;
  return (uint16_t)(rounded >> 16);
}

struct CaseSpec {
  float temp;
  float pres;
  float freq;
  bool mask;
  bool counts;
};

static bool case_spec(const std::string& base, CaseSpec& cs) {
  if (base == "plain-t1")          cs = {1.0f, 0.0f, 0.0f, false, false};
  else if (base == "plain-t07")    cs = {0.7f, 0.0f, 0.0f, false, false};
  else if (base == "greedy")       cs = {0.0f, 0.0f, 0.0f, false, false};
  else if (base == "penalty")      cs = {1.0f, 0.5f, 0.3f, false, true};
  else if (base == "masked")       cs = {1.0f, 0.0f, 0.0f, true,  false};
  else if (base == "mask-penalty") cs = {1.0f, 0.5f, 0.3f, true, true};
  else if (base == "spec-k4")      cs = {1.0f, 0.0f, 0.0f, false, false};
  else return false;
  return true;
}

struct Shape {
  std::string base;
  int batch_rows = 1;
  int width = 1;
  int eff_rows() const { return batch_rows * width; }
};

static bool parse_case(const std::string& token, Shape& sh) {
  std::string s = token;
  const size_t colon = s.find(':');
  std::string shape = "";
  if (colon != std::string::npos) {
    shape = s.substr(colon + 1);
    s = s.substr(0, colon);
  }
  sh.base = s;
  CaseSpec cs;
  if (!case_spec(sh.base, cs)) return false;
  if (!shape.empty()) {
    int r = 0, w = 0;
    if (sscanf(shape.c_str(), "r%dw%d", &r, &w) == 2) { sh.batch_rows = r; sh.width = w; }
    else if (sscanf(shape.c_str(), "r%d", &r) == 1) { sh.batch_rows = r; }
    else if (sscanf(shape.c_str(), "w%d", &w) == 1) { sh.width = w; }
    else return false;
  }
  return sh.batch_rows >= 1 && sh.width >= 1;
}

struct HostData {
  int eff_rows = 0;
  std::vector<float> temperature, presence, frequency; // [eff_rows]
  std::vector<uint16_t> logits;                        // [V, eff_rows] row-major
  std::vector<int32_t> counts;                         // [eff_rows, V] row-major
  std::vector<int32_t> valid_per_row;
  uint64_t seed = 0x5EED1234ULL;
};

static HostData generate(const Shape& sh) {
  CaseSpec cs;
  case_spec(sh.base, cs);
  const int eff = sh.eff_rows();
  HostData hd;
  hd.eff_rows = eff;
  hd.temperature.assign(eff, cs.temp);
  hd.presence.assign(eff, cs.pres);
  hd.frequency.assign(eff, cs.freq);
  hd.logits.resize((size_t)V * eff);
  hd.counts.assign((size_t)eff * V, 0);

  const int valid_start = V / 10; // ~90% contiguous suffix masked
  Rng rng_logits(hd.seed);
  for (int v = 0; v < V; ++v) {
    for (int r = 0; r < eff; ++r) {
      float f = rng_logits.n() * 2.5f;
      if (cs.mask && v >= valid_start) f = -INFINITY;
      hd.logits[(size_t)v * eff + r] = f32_to_bf16_rn(f);
    }
  }
  Rng rng_counts(hd.seed ^ 0xC0FFEEULL);
  if (cs.counts) {
    for (int r = 0; r < eff; ++r)
      for (int v = 0; v < V; ++v)
        hd.counts[(size_t)r * V + v] = (int32_t)(rng_counts.next() % 5u); // 0..4
  }
  hd.valid_per_row.assign(eff, cs.mask ? (V - valid_start) : V);
  return hd;
}

// ---------------------------------------------------------------------------
// CUDA buffers + run
// ---------------------------------------------------------------------------

struct DeviceBufs {
  uint16_t* logits = nullptr;
  int32_t* counts = nullptr;
  float* temperature = nullptr;
  float* presence = nullptr;
  float* frequency = nullptr;
  float* pmax = nullptr;
  float* psum = nullptr;
  float* ptop_v = nullptr;
  int32_t* ptop_i = nullptr;
  float* top_values = nullptr;
  int32_t* top_ids = nullptr;
  float* lse = nullptr;
  int32_t* chosen = nullptr;
  int* invalid = nullptr;
};

static void alloc_bufs(DeviceBufs& d, const HostData& hd) {
  const int eff = hd.eff_rows;
  const size_t parts = (size_t)eff * SPLIT;
  CK(cudaMalloc(&d.logits, (size_t)V * eff * sizeof(uint16_t)));
  CK(cudaMalloc(&d.counts, (size_t)eff * V * sizeof(int32_t)));
  CK(cudaMalloc(&d.temperature, eff * sizeof(float)));
  CK(cudaMalloc(&d.presence, eff * sizeof(float)));
  CK(cudaMalloc(&d.frequency, eff * sizeof(float)));
  CK(cudaMalloc(&d.pmax, parts * sizeof(float)));
  CK(cudaMalloc(&d.psum, parts * sizeof(float)));
  CK(cudaMalloc(&d.ptop_v, parts * K * sizeof(float)));
  CK(cudaMalloc(&d.ptop_i, parts * K * sizeof(int32_t)));
  CK(cudaMalloc(&d.top_values, (size_t)K * eff * sizeof(float)));
  CK(cudaMalloc(&d.top_ids, (size_t)K * eff * sizeof(int32_t)));
  CK(cudaMalloc(&d.lse, eff * sizeof(float)));
  CK(cudaMalloc(&d.chosen, eff * sizeof(int32_t)));
  CK(cudaMalloc(&d.invalid, sizeof(int)));
}

static void upload_bufs(DeviceBufs& d, const HostData& hd) {
  const int eff = hd.eff_rows;
  CK(cudaMemcpy(d.logits, hd.logits.data(), (size_t)V * eff * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.counts, hd.counts.data(), (size_t)eff * V * sizeof(int32_t), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.temperature, hd.temperature.data(), eff * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.presence, hd.presence.data(), eff * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d.frequency, hd.frequency.data(), eff * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemset(d.invalid, 0, sizeof(int)));
}

static void free_bufs(DeviceBufs& d) {
  cudaFree(d.logits); cudaFree(d.counts); cudaFree(d.temperature); cudaFree(d.presence);
  cudaFree(d.frequency); cudaFree(d.pmax); cudaFree(d.psum); cudaFree(d.ptop_v);
  cudaFree(d.ptop_i); cudaFree(d.top_values); cudaFree(d.top_ids); cudaFree(d.lse);
  cudaFree(d.chosen); cudaFree(d.invalid);
}

static void launch(const DeviceBufs& d, int eff) {
  dim3 grid_p(SPLIT, eff);
  partials_kernel<<<grid_p, THREADS>>>(d.logits, d.counts, d.temperature, d.presence,
                                       d.frequency, V, eff, SPLIT, d.pmax, d.psum,
                                       d.ptop_v, d.ptop_i);
  CK(cudaGetLastError());
  combine_kernel<<<eff, THREADS>>>(d.pmax, d.psum, d.ptop_v, d.ptop_i, eff, SPLIT,
                                   d.top_values, d.top_ids, d.lse, d.chosen, d.invalid);
  CK(cudaGetLastError());
}

static void check_invalid(const DeviceBufs& d) {
  int h = 0;
  CK(cudaMemcpy(&h, d.invalid, sizeof(int), cudaMemcpyDeviceToHost));
  if (h) {
    fprintf(stderr, "FATAL: invalid row (max scaled logit == -inf; all tokens masked)\n");
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

static void write_raw(const std::string& dir, const char* name, const void* p, size_t bytes) {
  const std::string path = dir + "/" + name;
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  if (bytes && fwrite(p, 1, bytes, f) != bytes) { fprintf(stderr, "short write %s\n", path.c_str()); exit(1); }
  fclose(f);
}

static void write_meta(const std::string& dir, const Shape& sh, const HostData& hd) {
  const std::string path = dir + "/meta.txt";
  FILE* f = fopen(path.c_str(), "w");
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
  fprintf(f, "case %s\n", sh.base.c_str());
  fprintf(f, "V %d\n", V);
  fprintf(f, "rows %d\n", hd.eff_rows);
  fprintf(f, "batch_rows %d\n", sh.batch_rows);
  fprintf(f, "width %d\n", sh.width);
  fprintf(f, "split %d\n", SPLIT);
  fprintf(f, "logits_shape V rows\n");
  fprintf(f, "counts_shape rows V\n");
  fprintf(f, "seed %llu\n", (unsigned long long)hd.seed);
  fprintf(f, "valid_per_row");
  for (int r = 0; r < hd.eff_rows; ++r) fprintf(f, " %d", hd.valid_per_row[r]);
  fprintf(f, "\n");
  fprintf(f, "temperature");
  for (int r = 0; r < hd.eff_rows; ++r) fprintf(f, " %.9g", hd.temperature[r]);
  fprintf(f, "\n");
  fprintf(f, "presence");
  for (int r = 0; r < hd.eff_rows; ++r) fprintf(f, " %.9g", hd.presence[r]);
  fprintf(f, "\n");
  fprintf(f, "frequency");
  for (int r = 0; r < hd.eff_rows; ++r) fprintf(f, " %.9g", hd.frequency[r]);
  fprintf(f, "\n");
  fclose(f);
}

static int do_gen(const Shape& sh, const std::string& dir) {
  HostData hd = generate(sh);
  const int eff = hd.eff_rows;
  DeviceBufs d;
  alloc_bufs(d, hd);
  upload_bufs(d, hd);
  launch(d, eff);
  CK(cudaDeviceSynchronize());
  check_invalid(d);

  std::vector<float> h_topv((size_t)K * eff);
  std::vector<int32_t> h_topi((size_t)K * eff);
  std::vector<float> h_lse(eff);
  std::vector<int32_t> h_chosen(eff);
  CK(cudaMemcpy(h_topv.data(), d.top_values, (size_t)K * eff * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_topi.data(), d.top_ids, (size_t)K * eff * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_lse.data(), d.lse, eff * sizeof(float), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(h_chosen.data(), d.chosen, eff * sizeof(int32_t), cudaMemcpyDeviceToHost));

  write_raw(dir, "logits.bf16", hd.logits.data(), hd.logits.size() * sizeof(uint16_t));
  write_raw(dir, "counts.i32", hd.counts.data(), hd.counts.size() * sizeof(int32_t));
  write_raw(dir, "top_values.f32", h_topv.data(), h_topv.size() * sizeof(float));
  write_raw(dir, "top_ids.i32", h_topi.data(), h_topi.size() * sizeof(int32_t));
  write_raw(dir, "lse.f32", h_lse.data(), h_lse.size() * sizeof(float));
  write_raw(dir, "chosen.i32", h_chosen.data(), h_chosen.size() * sizeof(int32_t));
  write_meta(dir, sh, hd);

  free_bufs(d);
  printf("GEN case=%s batch_rows=%d width=%d eff_rows=%d dir=%s ok\n",
         sh.base.c_str(), sh.batch_rows, sh.width, eff, dir.c_str());
  return 0;
}

static int do_bench(const Shape& sh) {
  HostData hd = generate(sh);
  const int eff = hd.eff_rows;
  DeviceBufs d;
  alloc_bufs(d, hd);
  upload_bufs(d, hd);

  const int warmup = 10;
  const int iters = 100;
  for (int i = 0; i < warmup; ++i) launch(d, eff);
  CK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CK(cudaEventCreate(&start));
  CK(cudaEventCreate(&stop));
  CK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) launch(d, eff);
  CK(cudaEventRecord(stop));
  CK(cudaEventSynchronize(stop));
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, start, stop));
  const double mean_us = (double)ms * 1000.0 / iters;

  printf("BENCH case=%s batch_rows=%d width=%d eff_rows=%d mean_us=%.3f\n",
         sh.base.c_str(), sh.batch_rows, sh.width, eff, mean_us);

  CK(cudaEventDestroy(start));
  CK(cudaEventDestroy(stop));
  free_bufs(d);
  return 0;
}

static void usage(const char* argv0) {
  fprintf(stderr,
          "usage:\n"
          "  %s --gen   <case>[:rN[wM]] <outdir>\n"
          "  %s --bench <case>[:rN[wM]]\n"
          "cases: plain-t1 plain-t07 greedy penalty masked mask-penalty spec-k4\n",
          argv0, argv0);
}

int main(int argc, char** argv) {
  if (argc < 3) { usage(argv[0]); return 2; }
  const std::string mode = argv[1];
  Shape sh;
  if (!parse_case(argv[2], sh)) { fprintf(stderr, "bad case: %s\n", argv[2]); return 2; }

  if (mode == "--gen") {
    if (argc != 4) { usage(argv[0]); return 2; }
    return do_gen(sh, argv[3]);
  } else if (mode == "--bench") {
    if (argc != 3) { usage(argv[0]); return 2; }
    return do_bench(sh);
  }
  usage(argv[0]);
  return 2;
}
