// Kernel A/B for the NVFP4 routed gate/up prefill stage (ticket #211, map #175). Times the
// production `sparse_moe_nvfp4_gate_up_launch` — activation quantisation + grouped W4A4
// `mma_nvfp4_e4m3` gate/up GEMM — over a token sweep at the registered 35B-A3B expert geometry
// (gate/up N=1024 K=2048), against the published Q4/Q5 sparse-MoE body baseline.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sparse_moe.h"

#include "ninfer_bench_common.h"
#include "ops/sparse_moe/nvfp4/sparse_moe_nvfp4_gate_up.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden       = 2048;
constexpr std::int32_t kExperts      = 256;
constexpr std::int32_t kTopK         = 8;
constexpr std::int32_t kIntermediate = 512;
constexpr std::int32_t kGateRows     = kIntermediate;
constexpr std::size_t kFlushBytes    = 256ULL << 20;

struct Options {
    std::vector<std::int32_t> t_sweep{20, 256, 512, 1024, 2048};
    int warmup   = 10;
    int repeat   = 50;
    std::string csv_out;
};

std::vector<std::int32_t> parse_t_sweep(std::string_view raw) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        const long value = std::stol(token);
        if (value <= 0 || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("--t-sweep values must be positive int32");
        }
        result.push_back(static_cast<std::int32_t>(value));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("--t-sweep must not be empty"); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--t-sweep") {
            options.t_sweep = parse_t_sweep(next("--t-sweep"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup")));
        } else if (argument == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat")));
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out");
        } else if (argument == "--help" || argument == "-h") {
            std::printf("Usage: %s [--t-sweep 20,256,...] [--warmup N] [--repeat N] "
                        "[--csv-out PATH]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (*std::min_element(options.t_sweep.begin(), options.t_sweep.end()) < 20) {
        throw std::invalid_argument("--t-sweep must stay in the nvfp4 prefill domain (T >= 20)");
    }
    return options;
}

struct HostRoute {
    std::vector<int> packed_token;
    std::vector<int> expert_offsets;
    std::vector<int> job_experts;
    std::vector<int> job_columns;
    int assignments = 0;
    int jobs        = 0;
};

DeviceBuffer to_device_i32(const std::vector<int>& host) {
    DeviceBuffer buffer(host.size() * sizeof(int));
    CUDA_CHECK(cudaMemcpy(buffer.p, host.data(), buffer.bytes, cudaMemcpyHostToDevice));
    return buffer;
}

HostRoute build_route(std::int32_t tokens) {
    HostRoute route;
    std::vector<std::vector<int>> tokens_by_expert(kExperts);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t rank = 0; rank < kTopK; ++rank) {
            tokens_by_expert[(token * 5 + rank * 29 + 7) % kExperts].push_back(token);
        }
    }
    route.assignments = tokens * kTopK;
    route.packed_token.resize(route.assignments);
    route.expert_offsets.assign(kExperts + 1, 0);
    for (int expert = 0; expert < kExperts; ++expert) {
        route.expert_offsets[expert + 1] =
            route.expert_offsets[expert] + static_cast<int>(tokens_by_expert[expert].size());
        for (int index = 0; index < static_cast<int>(tokens_by_expert[expert].size()); ++index) {
            route.packed_token[route.expert_offsets[expert] + index] =
                tokens_by_expert[expert][index];
        }
        const int count = static_cast<int>(tokens_by_expert[expert].size());
        for (int column = 0; column < count; column += 64) {
            route.job_experts.push_back(expert);
            route.job_columns.push_back(column);
        }
    }
    route.jobs = static_cast<int>(route.job_experts.size());
    return route;
}

struct Point {
    std::int32_t tokens = 0;
    DeviceBuffer packed_token;
    DeviceBuffer expert_offsets;
    DeviceBuffer job_experts;
    DeviceBuffer job_columns;
    DeviceBuffer job_count;
    DeviceBuffer plane_codes;
    DeviceBuffer plane_scales;
    DeviceBuffer output;
    int assignments = 0;
};

Point make_point(std::int32_t tokens) {
    const HostRoute route = build_route(tokens);
    Point point;
    point.tokens      = tokens;
    point.assignments = route.assignments;
    point.packed_token   = to_device_i32(route.packed_token);
    point.expert_offsets = to_device_i32(route.expert_offsets);
    point.job_experts    = to_device_i32(route.job_experts);
    point.job_columns    = to_device_i32(route.job_columns);
    point.job_count      = to_device_i32({route.jobs, 0});
    point.plane_codes = DeviceBuffer(std::size_t(tokens) * (kHidden / 2));
    point.plane_scales = DeviceBuffer(std::size_t(tokens) * (kHidden / 16));
    point.output = DeviceBuffer(std::size_t(kIntermediate) * route.assignments * sizeof(std::uint16_t));
    return point;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::int32_t max_t =
            *std::max_element(options.t_sweep.begin(), options.t_sweep.end());

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        bench::L2FlushBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_t, 211U);

        // One complete nvfp4 parent per expert projection, as the routed bank binds them.
        std::vector<bench::PackedQuantizedWeight> parents;
        parents.reserve(2 * kExperts);
        for (int expert = 0; expert < kExperts; ++expert) {
            parents.push_back(bench::make_nvfp4_weight(kGateRows, kHidden, 0x211u + expert * 2));
            parents.push_back(bench::make_nvfp4_weight(kGateRows, kHidden, 0x212u + expert * 2));
        }
        ops::SparseMoeWeights weights{};
        weights.routed_gate_up.qtype = QType::NVFP4;
        weights.routed_down.qtype    = QType::NVFP4;
        weights.routed_gate_up_experts.reserve(parents.size());
        for (const bench::PackedQuantizedWeight& parent : parents) {
            weights.routed_gate_up_experts.push_back(parent.weight);
        }

        const std::uint64_t weight_bytes =
            2ull * kExperts * parents.front().model_weight_bytes();
        std::printf("%-16s %8s %8s %6s %11s %11s %11s %9s %10s\n", "op", "N", "K", "T", "median_us",
                    "min_us", "p95_us", "eff_GB/s", "TFLOP/s");

        std::vector<Point> points;
        points.reserve(options.t_sweep.size());
        for (const std::int32_t tokens : options.t_sweep) { points.push_back(make_point(tokens)); }

        std::ofstream csv;
        if (!options.csv_out.empty()) {
            const std::filesystem::path path(options.csv_out);
            if (!path.parent_path().empty()) {
                std::filesystem::create_directories(path.parent_path());
            }
            csv.open(path);
            if (!csv) { throw std::runtime_error("failed to open CSV: " + options.csv_out); }
            csv << "op,weight_type,N,K,T,weight_bytes,median_us,min_us,p95_us,effective_gbs,"
                   "useful_tflops,warmup,repeat,flush_bytes\n";
        }

        for (Point& point : points) {
            const auto launch = [&](cudaStream_t launch_stream) {
                Tensor x(input.p, DType::BF16, {kHidden, point.tokens});
                ops::detail::sparse_moe_nvfp4_gate_up_launch(
                    x, weights, static_cast<const int*>(point.packed_token.p),
                    static_cast<const int*>(point.expert_offsets.p),
                    static_cast<const int*>(point.job_experts.p),
                    static_cast<const int*>(point.job_columns.p),
                    static_cast<const int*>(point.job_count.p),
                    static_cast<std::uint8_t*>(point.plane_codes.p),
                    static_cast<std::uint8_t*>(point.plane_scales.p),
                    static_cast<__nv_bfloat16*>(point.output.p), launch_stream);
            };
            const bench::ColdTiming timing = bench::measure_cold_launch(
                launch, flush, stream, options.warmup, options.repeat);
            const double seconds = timing.median_us * 1.0e-6;
            const double flops =
                2.0 * static_cast<double>(point.assignments) * 2.0 * kGateRows * kHidden;
            const double useful_tflops = flops / seconds / 1.0e12;
            const double effective_gbs  = static_cast<double>(weight_bytes) / seconds / 1.0e9;
            std::printf("%-16s %8d %8d %6d %11.3f %11.3f %11.3f %9.1f %10.2f\n",
                        "nvfp4_gate_up", 2 * kGateRows, kHidden, point.tokens, timing.median_us,
                        timing.min_us, timing.p95_us, effective_gbs, useful_tflops);
            if (csv) {
                csv << "sparse_moe_gate_up,NVFP4," << 2 * kGateRows << ',' << kHidden << ','
                    << point.tokens << ',' << weight_bytes << ',' << timing.median_us << ','
                    << timing.min_us << ',' << timing.p95_us << ',' << effective_gbs << ','
                    << useful_tflops << ',' << options.warmup << ',' << options.repeat << ','
                    << kFlushBytes << '\n';
            }
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_nvfp4_sparse_moe_gate_up_bench: %s\n", error.what());
        return 1;
    }
}
