#include "models/qwen3_5/frontend/grammar/grammar_runtime.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

std::shared_ptr<const CompiledGrammar>
require_grammar(std::shared_ptr<const CompiledGrammar> grammar) {
    if (grammar == nullptr) {
        throw std::invalid_argument("grammar runtime needs a compiled grammar");
    }
    return grammar;
}

void check_plan_spans(std::span<const int> tokens, std::span<const std::uint8_t> regions) {
    if (tokens.size() != regions.size()) {
        throw std::invalid_argument("grammar runtime plan spans must have the same length");
    }
}

} // namespace

GrammarRuntime::GrammarRuntime(std::shared_ptr<const CompiledGrammar> grammar)
    : grammar_(require_grammar(std::move(grammar))), state_(grammar_->initial_state()) {}

GrammarRuntime::GrammarRuntime(GrammarRuntime&&) noexcept            = default;
GrammarRuntime& GrammarRuntime::operator=(GrammarRuntime&&) noexcept = default;
GrammarRuntime::~GrammarRuntime()                                    = default;

void GrammarRuntime::fill_round_rows(std::span<const int> tokens,
                                     std::span<const std::uint8_t> regions,
                                     std::span<std::uint32_t> rows, std::size_t words) const {
    check_plan_spans(tokens, regions);
    const std::size_t columns = regions.size();
    if (words == 0 || rows.size() < columns * words) {
        throw std::invalid_argument("grammar runtime row buffer is too short");
    }
    const std::size_t row_words = (static_cast<std::size_t>(grammar_->token_count()) + 31U) / 32U;
    if (words < row_words) {
        throw std::invalid_argument("grammar runtime row width does not cover the token domain");
    }

    GrammarState preview = state_;
    bool reachable       = true;
    for (std::size_t column = 0; column < columns; ++column) {
        const std::span<std::uint32_t> row = rows.subspan(column * words, words);
        if (regions[column] != 0 || !reachable) {
            std::fill(row.begin(), row.end(), 0xFFFFFFFFU);
            continue;
        }
        const GrammarMaskRow masked = grammar_->row_for(preview);
        std::copy(masked.begin(), masked.end(), row.begin());
        // Bits past the token domain are not tokens: nothing is forbidden there.
        std::fill(row.begin() + static_cast<std::ptrdiff_t>(masked.size()), row.end(), 0xFFFFFFFFU);
        if (column + 1 == columns) { continue; }
        if (tokens[column] == kUnknownConstraintToken || !preview.accept(tokens[column])) {
            reachable = false;
        }
    }
}

bool GrammarRuntime::commit(std::span<const int> accepted, std::span<const std::uint8_t> regions) {
    check_plan_spans(accepted, regions);
    GrammarState next = state_;
    for (std::size_t column = 0; column < accepted.size(); ++column) {
        if (regions[column] != 0) { continue; }
        if (accepted[column] == kUnknownConstraintToken || !next.accept(accepted[column])) {
            return false;
        }
    }
    state_ = std::move(next);
    return true;
}

const CompiledGrammar& GrammarRuntime::grammar() const noexcept { return *grammar_; }

bool GrammarRuntime::can_end() const noexcept { return state_.can_end(); }

} // namespace ninfer::models::qwen3_5::frontend
