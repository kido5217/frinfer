#pragma once

#include "ninfer/ops/moe.h"
#include "ops/moe/kernels/cuda/groupwise_int/launch.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::moe_cuda::fork_nvfp4 {

// The fork 35B-A3B NVFP4 route (h2048/i512/e256/k8). Routed experts are complete encoded
// BlockScaleK16M128x4 parents in one affine device bank (gate rows first, then up rows, 2I rows
// per expert). The grouped stage runs the ported W4A4 persistent kernels
// (ops/moe/nvfp4/sparse_moe_nvfp4_{gate_up,down}.h) at T >= 20; below it the A16 SIMT pair
// decodes the same parents against the BF16 hidden state and the FP32 SwiGLU activation.
// The shared expert is a Q8 RowSplit SwiGLU and rides the groupwise_int family; this launch
// composes it through the caller-supplied shared storage.
inline constexpr int kHidden = 2048, kIntermediate = 512, kExperts = 256, kTopK = 8;
inline constexpr int kGroupedFirstTokens = 20;

struct Plan {
    // One compact field set; the token boundary is already resolved at plan selection.
    bool grouped = false;
    int queued_ctas_per_sm = 32;
    groupwise_int::SharedPlan shared;
};

struct Storage {
    int* expert_offsets = nullptr;   // [kExperts + 1]
    int* counts         = nullptr;   // [kExperts] histogram inputs (grouped route)
    int* cursors        = nullptr;   // [kExperts] fill cursors (grouped route)
    int* job_experts    = nullptr;   // [max_jobs]
    int* job_columns    = nullptr;   // [max_jobs]
    int* job_count      = nullptr;   // [2]: [0] job count, [1] touched experts
    int* packed_token   = nullptr;   // [assignments]: routed token of each packed column
    int* inverse        = nullptr;   // [assignments]: packed column per assignment, -1 unselected
    std::uint8_t* input_codes   = nullptr;  // [T * kHidden / 2]
    void* input_scales          = nullptr;  // [T * kHidden / 16]
    std::uint8_t* product_codes = nullptr;  // [assignments * kIntermediate / 2]
    void* product_scales        = nullptr;  // [assignments * kIntermediate / 16]
    // gate/up writes the SwiGLU activation as [kIntermediate, assignments]; the down stage
    // quantises it into the product plane and then overwrites this span with the grouped
    // contributions [kHidden, assignments]. The two stages are stream-ordered, so one span
    // serves both.
    void* routed_io = nullptr;
    float* vector_activation = nullptr;  // [kIntermediate, assignments] on the A16 stage
    float* routed_sum        = nullptr;  // [kHidden, tokens] when the shared term composes
};

template <class Arena>
Storage allocate(Arena& arena, const Plan& plan, int tokens, bool routed, bool shared) {
    Storage out;
    if (!routed) { return out; }
    const int assignments = tokens * kTopK;
    const auto integers = [&](int count) {
        return static_cast<int*>(arena.alloc(DType::I32, {count}).data);
    };
    const auto bytes = [&](std::size_t count) {
        return static_cast<std::uint8_t*>(arena.alloc_bytes(count).data);
    };
    if (plan.grouped) {
        out.expert_offsets = integers(kExperts + 1);
        out.counts         = integers(kExperts);
        out.cursors        = integers(kExperts);
        // One job per 64-column tile of each expert plus one tail job per touched expert.
        const int max_jobs = assignments / 64 + kExperts;
        out.job_experts    = integers(max_jobs);
        out.job_columns    = integers(max_jobs);
        out.job_count      = integers(2);
        out.packed_token   = integers(assignments);
        out.inverse        = integers(assignments);
        out.input_codes    = bytes(std::size_t(tokens) * (kHidden / 2));
        out.input_scales   = bytes(std::size_t(tokens) * (kHidden / 16));
        out.product_codes  = bytes(std::size_t(assignments) * (kIntermediate / 2));
        out.product_scales = bytes(std::size_t(assignments) * (kIntermediate / 16));
        out.routed_io = bytes(std::size_t(assignments) * kHidden * sizeof(std::uint16_t));
        if (shared) {
            out.routed_sum =
                static_cast<float*>(arena.alloc(DType::FP32, {kHidden, tokens}).data);
        }
    } else {
        out.vector_activation =
            static_cast<float*>(arena.alloc(DType::FP32, {kIntermediate, assignments}).data);
        if (shared) {
            out.routed_sum =
                static_cast<float*>(arena.alloc(DType::FP32, {kHidden, tokens}).data);
        }
    }
    return out;
}

// Runs the routed experts and, when a shared term is supplied, composes it. The routed part
// finishes into shared_storage.routed_sum whenever a shared term follows, so the groupwise
// shared-down pass publishes destination once (mode semantics included). `shared_storage`
// must be a groupwise_int allocation for the same token count whenever `shared` is non-null.
void launch(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
            ExpertSubset subset, const ScaledSwiGLUTerm* shared, MoeEpilogue mode,
            Tensor& destination, const Plan& plan, const Storage& storage,
            groupwise_int::ExpertStorage shared_storage, CudaExecutionView execution);

} // namespace ninfer::ops::detail::moe_cuda::fork_nvfp4
