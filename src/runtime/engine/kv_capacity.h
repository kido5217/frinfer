#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::runtime {

[[nodiscard]] KvCapacityResolution resolve_kv_capacity(const KvCapacityPolicy& policy,
                                                       const SequenceCapacityCurve& curve,
                                                       std::size_t available_runtime_bytes);

// Fork contract: active-request preemption is pinned off, so a resolved Main KV pool must be the
// curve's full worst-case pool (curve.maximum_main_page_groups), which covers max_concurrency lanes
// at full context. Throws std::invalid_argument naming the required and supplied capacity when a
// resolution undersizes it.
void require_full_resident_capacity(const SequenceCapacityCurve& curve, std::uint32_t resolved_pages);

} // namespace ninfer::runtime
