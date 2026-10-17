#pragma once

#include "core/device.h"
#include "ninfer/ops/rope.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rope_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                   CudaExecutionView execution);

void rope_dispatch(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
                   CudaExecutionView execution);

// YaRN extension dispatch (pair and single-tensor forms): identical validation and geometry
// tuning as the power-law path; the kernel reads the device-resident table and applies mscale.
void rope_yarn_dispatch(const Tensor& positions, int rotary_dim, float theta,
                        const YarnScale& yarn, Tensor& q, Tensor& k, CudaExecutionView execution);

void rope_yarn_dispatch(const Tensor& positions, int rotary_dim, float theta,
                        const YarnScale& yarn, Tensor& x, CudaExecutionView execution);

} // namespace ninfer::ops::detail
