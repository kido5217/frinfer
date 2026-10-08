#pragma once

#include "models/load_options.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5 {
struct RopeConfig;
} // namespace ninfer::models::qwen3_5

namespace ninfer::models::qwen3_5::detail {

// Fork YaRN context-extension policy. The extension is active exactly when the planned logical
// capacity exceeds the model's native position capacity. `persistent_layout` (host-side table
// construction) and `validate_target_options` (startup rejection) share this single predicate so
// the built table and the rejection agree on the boundary. At or below native the extension is
// inactive and the table is empty, leaving the power-law rope path structurally unchanged.
[[nodiscard]] constexpr bool
yarn_extension_active(std::uint32_t capacity, std::uint32_t native_position_capacity) noexcept {
    return native_position_capacity > 0 && capacity > native_position_capacity;
}

// Host-side YaRN table for a plan under the extension policy. Inactive and empty at or below the
// native capacity (byte-identical power-law path); above it, the fp64 per-pair frequencies and the
// attention magnitude produced by `ops::compute_rope_yarn_table`. `active` mirrors
// `yarn_extension_active(capacity, native_position_capacity)`.
struct YarnExtensionPlan {
    bool active = false;
    std::vector<float> inv_freq;
    float mscale = 1.0f;
};

[[nodiscard]] YarnExtensionPlan plan_yarn_extension(const RopeConfig* rope,
                                                    std::uint32_t native_position_capacity,
                                                    std::uint32_t capacity);

// Startup guard: a masked (DFlash) draft backend keeps its own native position bound and cannot
// ride the text YaRN table, so an active extension with that backend is rejected before any
// allocation. A non-masking backend (including MTP) is allowed.
void validate_yarn_extension(std::uint32_t max_context, std::uint32_t native_position_capacity,
                             SpeculativeBackend backend);

} // namespace ninfer::models::qwen3_5::detail
