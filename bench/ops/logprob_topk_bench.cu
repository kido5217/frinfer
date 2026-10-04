// Qualification benchmark for the bounded top-k logprob gather over sampler logits.
//
//   ./ninfer_logprob_topk_bench
//   ./ninfer_logprob_topk_bench --rows 8
//
// Shapes match the decode-round domains: one column per ordinary request and one column per
// speculative verify position (rows = width*batch). The reported step share uses the 13.7 ms
// decode step the evidence gate measured at V=151936.
#include "ninfer/ops/logprob_topk.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kVocab      = 151936;
constexpr double kDecodeStepMs     = 13.7;

int parse_int(std::string_view value, const char* name) {
    try {
        std::size_t parsed = 0;
        const int out      = std::stoi(std::string(value), &parsed);
        if (parsed != value.size()) { throw std::invalid_argument("trailing"); }
        return out;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(name) + " expects an integer");
    }
}

void run_rows(std::int32_t rows) {
    DeviceBuffer logits = make_bf16(static_cast<std::size_t>(kVocab) * rows, 101U, -6.0F, 6.0F);
    DeviceBuffer configs = make_zeros(static_cast<std::size_t>(rows) * sizeof(ops::SamplingConfig));
    DeviceBuffer top_ids(static_cast<std::size_t>(ops::kLogprobTopK) * rows * sizeof(std::int32_t));
    DeviceBuffer top_values(static_cast<std::size_t>(ops::kLogprobTopK) * rows * sizeof(float));
    DeviceBuffer lse(static_cast<std::size_t>(rows) * sizeof(float));
    DeviceBuffer active(sizeof(std::int32_t));
    const std::int32_t one = 1;
    active.copy_from_host(&one, sizeof(one));

    Tensor logits_tensor(logits.p, DType::BF16, {kVocab, rows});
    Tensor ids_tensor(top_ids.p, DType::I32, {ops::kLogprobTopK, rows});
    Tensor values_tensor(top_values.p, DType::FP32, {ops::kLogprobTopK, rows});
    Tensor lse_tensor(lse.p, DType::FP32, {rows});
    Tensor active_tensor(active.p, DType::I32, {1});
    auto* config_pointer = reinterpret_cast<const ops::SamplingConfig*>(configs.p);

    WorkspaceArena workspace(ops::logprob_topk_workspace_capacity_bytes(kVocab, rows));
    const double bytes =
        static_cast<double>(kVocab) * rows * 2.0 * 2.0; // two BF16 vocabulary passes per row
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            ops::logprob_topk(logits_tensor, config_pointer, kVocab, ids_tensor, values_tensor,
                              lse_tensor, active_tensor, workspace, stream);
        },
        bytes);

    // The production Op runs inside the captured decode graph, where launch overhead is removed.
    // Time the same call replayed from a captured graph to bound the in-graph cost.
    const Result captured = bench_loop_prepared(
        [](cudaStream_t) {},
        [&](cudaStream_t stream) {
            ops::logprob_topk(logits_tensor, config_pointer, kVocab, ids_tensor, values_tensor,
                              lse_tensor, active_tensor, workspace, stream);
        },
        bytes, 20, 100);

    char label[96];
    std::snprintf(label, sizeof(label), "logprob_topk rows=%d V=%d", rows, kVocab);
    print_result(label, result);
    std::snprintf(label, sizeof(label), "logprob_topk rows=%d captured", rows);
    print_result(label, captured);
    std::printf("  rows=%d step share: %.3f%% standalone, %.3f%% captured of %.1f ms\n", rows,
                result.median_us * 1e-3 / kDecodeStepMs * 100.0,
                captured.median_us * 1e-3 / kDecodeStepMs * 100.0, kDecodeStepMs);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    try {
        std::vector<std::int32_t> shapes;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--rows") {
                if (i + 1 >= argc) { throw std::invalid_argument("--rows needs a value"); }
                shapes.push_back(parse_int(argv[++i], "--rows"));
            } else if (arg == "-h" || arg == "--help") {
                std::printf("usage: %s [--rows N]...\n", argv[0]);
                return 0;
            } else {
                throw std::invalid_argument("unknown argument");
            }
        }
        if (shapes.empty()) { shapes = {1, 8, 40}; }
        for (const std::int32_t rows : shapes) {
            if (rows <= 0) { throw std::invalid_argument("--rows must be positive"); }
            run_rows(rows);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ninfer_logprob_topk_bench: %s\n", e.what());
        return 2;
    }
    return 0;
}
