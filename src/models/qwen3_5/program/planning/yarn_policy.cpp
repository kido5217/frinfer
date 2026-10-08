#include "models/qwen3_5/program/planning/yarn_policy.h"

#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/rope.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::detail {

YarnExtensionPlan plan_yarn_extension(const RopeConfig* rope,
                                      std::uint32_t native_position_capacity,
                                      std::uint32_t capacity) {
    YarnExtensionPlan out;
    if (rope == nullptr || !yarn_extension_active(capacity, native_position_capacity)) {
        return out;
    }
    const double scale =
        static_cast<double>(capacity) / static_cast<double>(native_position_capacity);
    const auto table =
        ops::compute_rope_yarn_table(rope->rope_theta, execution::dimension(rope->rotary_dim),
                                     static_cast<std::int64_t>(native_position_capacity), scale);
    out.active   = true;
    out.inv_freq = std::move(table.inv_freq);
    out.mscale   = table.mscale;
    return out;
}

void validate_yarn_extension(std::uint32_t max_context, std::uint32_t native_position_capacity,
                             SpeculativeBackend backend) {
    if (yarn_extension_active(max_context, native_position_capacity) &&
        is_masked_draft_backend(backend)) {
        throw std::invalid_argument(
            "YaRN context extension (--max-context above max_position_embeddings) is not "
            "supported with a DFlash draft backend");
    }
}

} // namespace ninfer::models::qwen3_5::detail
