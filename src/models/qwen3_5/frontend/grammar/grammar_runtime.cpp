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

} // namespace

GrammarRuntime::GrammarRuntime(std::shared_ptr<const CompiledGrammar> grammar)
    : grammar_(require_grammar(std::move(grammar))), state_(grammar_->initial_state()) {}

GrammarRuntime::GrammarRuntime(GrammarRuntime&&) noexcept            = default;
GrammarRuntime& GrammarRuntime::operator=(GrammarRuntime&&) noexcept = default;
GrammarRuntime::~GrammarRuntime()                                    = default;

void GrammarRuntime::fill_round_rows(std::span<const int> tokens, std::span<std::uint32_t> rows,
                                     std::size_t words) const {
    const std::size_t columns = tokens.size();
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
        if (!reachable) {
            std::fill(row.begin(), row.end(), 0xFFFFFFFFU);
            continue;
        }
        const GrammarMaskRow masked = grammar_->row_for(preview);
        std::copy(masked.begin(), masked.end(), row.begin());
        // Words past the row width are padding: the consumer never inspects bits at or past the
        // token domain, so intra-word tail bits are left as the row produced them.
        std::fill(row.begin() + static_cast<std::ptrdiff_t>(masked.size()), row.end(), 0xFFFFFFFFU);
        if (column + 1 == columns) { continue; }
        if (tokens[column] == kUnknownConstraintToken || !preview.accept(tokens[column])) {
            reachable = false;
        }
    }
}

bool GrammarRuntime::commit(std::span<const int> accepted) {
    GrammarState next = state_;
    for (const int token : accepted) {
        if (token == kUnknownConstraintToken || !next.accept(token)) { return false; }
    }
    state_ = std::move(next);
    return true;
}

const CompiledGrammar& GrammarRuntime::grammar() const noexcept { return *grammar_; }

bool GrammarRuntime::can_end() const noexcept { return state_.can_end(); }

} // namespace ninfer::models::qwen3_5::frontend
