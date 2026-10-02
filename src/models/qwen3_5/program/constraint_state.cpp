#include "models/qwen3_5/program/constraint_state.h"

#include <algorithm>
#include <array>
#include <utility>

namespace ninfer::models::qwen3_5 {

void ConstraintState::attach(std::shared_ptr<const frontend::CompiledGrammar> grammar,
                             bool carries_reasoning) {
    planned_columns_   = 0;
    carries_reasoning_ = false;
    runtime_.reset();
    grammar_.reset();
    if (grammar == nullptr) { return; }
    carries_reasoning_ = carries_reasoning;
    grammar_           = std::move(grammar);
    runtime_           = std::make_unique<frontend::GrammarRuntime>(grammar_);
}

void ConstraintState::clear() noexcept {
    planned_columns_   = 0;
    carries_reasoning_ = false;
    runtime_.reset();
    grammar_.reset();
}

std::uint32_t ConstraintState::plan_round(std::span<const TokenId> drafts) {
    if (runtime_ == nullptr) {
        planned_columns_ = 0;
        return 0;
    }
    // Every column of a constrained lane is an answer column, and the round's columns are the
    // draft span plus the bonus column. The transport's column capacity bounds a defensive
    // overflow.
    planned_columns_ = static_cast<std::uint32_t>(
        std::min<std::size_t>(drafts.size() + 1U, MaskTransport::kColumnCapacity));
    return planned_columns_;
}

void ConstraintState::fill_row(std::span<const TokenId> drafts, std::uint32_t batch_position,
                               MaskTransport& transport) {
    if (runtime_ == nullptr || planned_columns_ == 0) { return; }
    std::array<int, MaskTransport::kColumnCapacity> tokens{};
    tokens.fill(frontend::kUnknownConstraintToken);
    const std::size_t known = std::min<std::size_t>(drafts.size(), planned_columns_ - 1U);
    for (std::size_t index = 0; index < known; ++index) { tokens[index] = drafts[index]; }
    std::span<std::uint32_t> block = transport.fill_block(planned_columns_);
    runtime_->fill_round_rows(std::span<const int>(tokens.data(), planned_columns_), block,
                              transport.words());
    transport.publish(batch_position, planned_columns_);
}

bool ConstraintState::commit(std::span<const int> accepted) {
    if (runtime_ == nullptr || planned_columns_ == 0) { return true; }
    return runtime_->commit(accepted);
}

bool ConstraintState::commit_forced(std::span<const int> forced) {
    if (runtime_ == nullptr || !carries_reasoning_) { return true; }
    return runtime_->commit(forced);
}

} // namespace ninfer::models::qwen3_5
