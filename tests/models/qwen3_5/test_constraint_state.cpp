// The per-lane constrained-round state (program/constraint_state.h): attach/clear, round
// planning, transport fill and commit, against a host-compiled grammar over a synthetic byte
// vocabulary.

#include "models/qwen3_5/frontend/grammar/grammar.h"
#include "models/qwen3_5/frontend/grammar/vocabulary.h"
#include "models/qwen3_5/program/constraint_state.h"
#include "models/qwen3_5/program/mask_transport.h"
#include "ninfer/types.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace frontend = ninfer::models::qwen3_5::frontend;
namespace q36      = ninfer::models::qwen3_5;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    if (ok) { return; }
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
}

bool row_has_bit(std::span<const std::uint32_t> row, int token) {
    const std::size_t index = static_cast<std::size_t>(token) / 32U;
    const std::uint32_t bit = 1U << (static_cast<std::uint32_t>(token) % 32U);
    return index < row.size() && (row[index] & bit) != 0;
}

// All 256 single-byte pieces (piece id = byte value) plus one end-of-generation id, so the
// module's plan/fill/commit path is exercised without a model artifact.
class ByteVocabulary final : public frontend::GrammarVocabulary {
public:
    [[nodiscard]] int token_count() const noexcept override { return 257; }

    [[nodiscard]] std::string_view piece(int id) const override {
        if (id < 0 || id > 256) {
            throw std::out_of_range("byte vocabulary piece id out of range");
        }
        if (id == 256) { return "<eos>"; }
        piece_[0] = static_cast<char>(id);
        return std::string_view(piece_.data(), 1);
    }

    [[nodiscard]] bool is_eog(int id) const override { return id == 256; }

    [[nodiscard]] std::vector<int> encode(std::string_view text) const override {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (const char byte : text) { ids.push_back(static_cast<unsigned char>(byte)); }
        return ids;
    }

private:
    mutable std::array<char, 1> piece_{' '};
};

} // namespace

int main() {
    constexpr int kNine          = '9';
    constexpr int kLetter        = 'a';
    constexpr int kEog           = 256;
    constexpr std::size_t kWords = (257 + 31) / 32; // the transport's row word count

    std::string error;
    const std::shared_ptr<const frontend::CompiledGrammar> grammar =
        frontend::CompiledGrammar::compile("root ::= \"9999\"", std::make_shared<ByteVocabulary>(),
                                           &error);
    check(grammar != nullptr, "a host grammar over the byte vocabulary did not compile");
    if (grammar == nullptr) {
        std::fprintf(stderr, "compile diagnostic: %s\n", error.c_str());
        return 1;
    }

    q36::MaskTransport transport(kWords); // host-only: the device side stays unattached
    q36::ConstraintState state;
    check(!state.constrained() && !state.carries_reasoning(),
          "a fresh lane reports constraint state");
    check(state.planned_columns() == 0, "a fresh lane has a planned round");

    state.attach(grammar, false);
    check(state.constrained(), "attach did not mark the lane constrained");
    check(!state.carries_reasoning(), "attach selected the thinking wrapper unexpectedly");

    // Prefill: one column at the grammar's initial state, which admits exactly the byte '9'.
    check(state.plan_round({}) == 1, "a constrained prefill did not plan one column");
    state.fill_row({}, 0, transport);
    const std::span<const std::uint32_t> initial_row = transport.staged_row(0, 0);
    check(row_has_bit(initial_row, kNine) && !row_has_bit(initial_row, kLetter),
          "the initial prefill row does not admit exactly the forced byte");

    // Commit the sampled token; an ungrammatical token fails closed without corrupting the state.
    const std::array<int, 1> nine{kNine};
    const std::array<int, 1> letter{kLetter};
    check(state.commit(nine), "committing the sampled token failed");
    check(!state.commit(letter), "an ungrammatical token was committed");
    check(state.commit(nine), "the failed commit advanced the grammar state");

    // Decode: the draft span plus the bonus column.
    const std::array<ninfer::TokenId, 2> drafts{static_cast<ninfer::TokenId>(kNine),
                                                static_cast<ninfer::TokenId>(kNine)};
    check(state.plan_round(drafts) == 3, "a decode round did not plan drafts + 1 columns");
    state.fill_row(drafts, 1, transport);
    const std::span<const std::uint32_t> second_row = transport.staged_row(0, 1);
    check(row_has_bit(second_row, kNine) && !row_has_bit(second_row, kLetter),
          "the second masked column does not continue the forced sequence");

    // The round licenses the remaining two nines and completes the grammar.
    const std::array<int, 2> accepted{kNine, kNine};
    check(state.commit(accepted), "committing the accepted span failed");
    check(state.plan_round({}) == 1, "the completed grammar did not plan one column");
    state.fill_row({}, 0, transport);
    const std::span<const std::uint32_t> complete_row = transport.staged_row(0, 0);
    check(row_has_bit(complete_row, kEog) && !row_has_bit(complete_row, kNine),
          "the completed grammar does not admit exactly the end-of-generation id");

    // The plan clamps to the transport's column capacity.
    std::array<ninfer::TokenId, 20> wide{};
    wide.fill(static_cast<ninfer::TokenId>(kNine));
    check(state.plan_round(wide) == q36::MaskTransport::kColumnCapacity,
          "an oversized draft span did not clamp to the transport column capacity");

    // The forced span only advances a wrapped lane: committing "9999" as forced control tokens
    // completes the grammar, so the next planned row admits only the end-of-generation id.
    state.attach(grammar, true);
    check(state.carries_reasoning(), "attach did not select the thinking wrapper");
    check(state.plan_round({}) == 1, "a wrapped lane did not plan its prefill column");
    const std::array<int, 4> forced_nines{kNine, kNine, kNine, kNine};
    check(state.commit_forced(forced_nines), "a wrapped lane did not commit its forced span");
    check(state.plan_round({}) == 1, "the forced span did not advance the grammar state");
    state.fill_row({}, 0, transport);
    const std::span<const std::uint32_t> forced_row = transport.staged_row(0, 0);
    check(row_has_bit(forced_row, kEog) && !row_has_bit(forced_row, kNine),
          "the forced span did not complete the grammar");

    // An unwrapped lane ignores the forced span.
    state.attach(grammar, false);
    check(state.plan_round({}) == 1, "a re-attached lane did not plan its prefill column");
    check(state.commit_forced(letter), "an unwrapped lane did not ignore its forced span");
    check(state.commit(nine), "an unwrapped lane's forced span advanced the grammar state");

    state.clear();
    check(!state.constrained() && state.planned_columns() == 0, "clear did not reset the lane");
    check(state.commit(nine), "a cleared lane did not ignore its commit");
    check(state.commit_forced(nine), "a cleared lane did not ignore its forced commit");

    if (g_failures != 0) {
        std::fprintf(stderr, "%d constraint-state checks failed\n", g_failures);
        return 1;
    }
    std::printf("constraint-state checks passed\n");
    return 0;
}
