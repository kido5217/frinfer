#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sparse_moe.h"
#include "ops/sparse_moe/nvfp4/sparse_moe_nvfp4_gate_up.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The per-expert weight bases the grouped W4A4 routed down GEMM reads: one complete encoded parent
// per expert (down is one [2048, 512] projection), carries its own weight divisor. Passed by value
// like the gate/up table, so the launch needs no device-resident weight table and stays valid
// across CUDA-graph capture and replay.
struct Nvfp4DownBases {
    const void* codes[kNvfp4RoutedExperts];
    const void* scales[kNvfp4RoutedExperts];
    float divisor[kNvfp4RoutedExperts];
};

// Builds the per-expert weight bases table from the bound nvfp4 routed down parents. Shared by the
// T>=20 prefill route and the T<20 SIMT route; the table is passed to the kernel by value.
[[nodiscard]] Nvfp4DownBases make_nvfp4_down_bases(const SparseMoeWeights& weights);

// Runs the nvfp4 routed down prefill stage: quantises the BF16 routed activation
// [assignments, 512] into a compact FP4 plane (one private activation divisor, as in the gate/up
// stage), then executes the grouped W4A4 down GEMM over the packed route, writing the BF16 grouped
// routed output [2048, assignments]. expert_offsets [257] and the route-job maps are the prefill
// route's packed-route maps (route_job_count[0] is the job count); activation_codes/scales are the
// compact FP4 plane scratch (assignments * 512/2 and assignments * 512/16 bytes).
// multiprocessor_count sizes the persistent grid cap (32 queued CTAs per SM), mirroring the packed
// prefill seam.
void sparse_moe_nvfp4_down_launch(const Tensor& routed_activation,
                                  const SparseMoeWeights& weights, const int* expert_offsets,
                                  const int* route_job_experts, const int* route_job_columns,
                                  const int* route_job_count, std::uint8_t* activation_codes,
                                  std::uint8_t* activation_scales,
                                  __nv_bfloat16* grouped_output,
                                  std::int32_t multiprocessor_count, cudaStream_t stream);

} // namespace ninfer::ops::detail
