#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sparse_moe.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr int kNvfp4RoutedExperts = 256;

// The per-expert weight bases the grouped W4A4 routed gate/up GEMM reads. One complete encoded
// parent per expert projection (gate and up are separate source tensors); the Op launches the
// kernel with this table by value, so it needs no device-resident weight table and stays valid
// across CUDA-graph capture and replay.
struct Nvfp4GateUpBases {
    const void* gate_codes[kNvfp4RoutedExperts];
    const void* gate_scales[kNvfp4RoutedExperts];
    const void* up_codes[kNvfp4RoutedExperts];
    const void* up_scales[kNvfp4RoutedExperts];
    float gate_divisor[kNvfp4RoutedExperts];
    float up_divisor[kNvfp4RoutedExperts];
};

// Runs the nvfp4 routed gate/up prefill stage: quantises the routed activation x into a compact
// FP4 plane shared by every expert a token routes to, then executes the grouped W4A4 gate/up GEMM
// over the packed route, writing the BF16 SwiGLU activation [512, assignments].
//
// expert_offsets [257], route_job_experts/columns [jobs] and route_job_count [2] are the prefill
// route's packed-route maps (route_job_count[0] is the job count). activation_codes/scales are the
// compact FP4 plane scratch (tokens * 2048/2 and tokens * 2048/16 bytes).
void sparse_moe_nvfp4_gate_up_launch(const Tensor& x, const SparseMoeWeights& weights,
                                     const int* packed_token, const int* expert_offsets,
                                     const int* route_job_experts,
                                     const int* route_job_columns, const int* route_job_count,
                                     std::uint8_t* activation_codes, std::uint8_t* activation_scales,
                                     __nv_bfloat16* routed_activation, cudaStream_t stream);

} // namespace ninfer::ops::detail
