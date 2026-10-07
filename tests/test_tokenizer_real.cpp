// Real-model tokenizer test (wayfinder map #175, ticket #192).
//
// The tokenizer surface `frinfer-tokenize` is built on: encode text to ids, decode ids back, and
// read one id's spelling and special-token status. A round trip is the property the whole debugging
// tool rests on — if encode/decode does not return the text, every answer it gives about token
// boundaries is wrong. The one caveat is normalisation: encoding normalises to NFC, so the round
// trip is the identity only on already-normalised text, and the normalised form otherwise.
//
// NINFER_TEST_ARTIFACT selects the artifact; without it the test skips (exit 77).

#include "ninfer/engine.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    if (ok) { return; }
    ++g_failures;
    std::cerr << "FAIL: " << what << (detail.empty() ? "" : ": " + detail) << '\n';
}

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.purpose         = ninfer::EnginePurpose::CausalScoring;
    options.max_context     = 2048;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    options.use_cuda_graph  = false;
    return options;
}

int exercise(const char* artifact) {
    ninfer::Engine engine(engine_options(artifact));

    // A corpus chosen for what breaks a tokenizer: multi-byte scripts, combining marks, code
    // whitespace, and padding that must survive verbatim.
    const std::vector<std::pair<std::string, std::string>> corpus = {
        {"plain", "The quick brown fox jumps over the lazy dog."},
        {"cjk", "日本語のテキストです。トークン境界を確認します。\n第二行。\n"},
        {"code", "def f(x):\n\treturn x ** 2  # note\n\n"},
        {"padding", "   leading and trailing whitespace   \n\n\n"},
        {"emoji", "emoji: \xF0\x9F\x9A\x80\xF0\x9F\x8E\x89 and combining: \xC3\xA9\n"},
        {"single", "a"},
        {"empty", ""},
        // Decomposed forms: encoding normalises to NFC, so this does not come back byte-identical.
        {"decomposed", "cafe\xCC\x81 na\xCC\x88ive\n"},
    };

    for (const auto& [name, text] : corpus) {
        const std::vector<ninfer::TokenId> ids = engine.tokenize_text(text);
        check(!ids.empty() || text.empty(),
              name + ": non-empty text tokenized to at least one id");
        // The concatenation of the per-id spellings is the decode, so the round trip returns the
        // text — byte-identical except where encoding's NFC normalisation changed it, which is
        // detected here by comparing sizes rather than by assuming a normalisation rule.
        const std::string decoded = engine.detokenize(ids);
        const bool normalised = decoded.size() != text.size();
        check(normalised || decoded == text,
              name + ": decode(encode(text)) returns the input text",
              "decoded bytes=" + std::to_string(decoded.size()) +
                  " original bytes=" + std::to_string(text.size()));
        if (normalised) {
            check(name == "decomposed",
                  "only the decomposed corpus entry is expected to change under NFC normalisation",
                  name + ": " + std::to_string(text.size()) + " -> " +
                      std::to_string(decoded.size()));
        }
        // Every id reads back individually, and the pieces in order are the decode.
        std::string by_hand;
        for (const ninfer::TokenId id : ids) {
            const ninfer::TokenPiece piece = engine.token_piece(id);
            by_hand += piece.text;
        }
        check(by_hand == decoded, name + ": per-id spellings concatenate to the decode");
        // No corpus entry names a control token, so suppressing them changes nothing here. That is
        // the invariant: raw text encoding adds no implicit control token, which is what makes a
        // count from this tool a count of the text and not of a framed prompt.
        check(engine.detokenize(ids, false) == engine.detokenize(ids, true),
              name + ": raw text encoding added an implicit control token");
    }

    // A framing marker is one id flagged special, and it has both views: the per-id spelling a
    // caller audits framing with, and a decode route where suppression is observable. Generated
    // content is rendered without the terminal marker, so reproducing a request's published text
    // from its ids is what --skip-special-tokens is for.
    for (const std::string marker : {std::string("<|") + "im_start" + "|>",
                                      std::string("<|") + "im_end" + "|>"}) {
        const std::vector<ninfer::TokenId> ids = engine.tokenize_text(marker);
        check(ids.size() == 1, marker + ": a named control token is a single id",
              "ids=" + std::to_string(ids.size()));
        if (ids.size() != 1) { continue; }
        const ninfer::TokenPiece piece = engine.token_piece(ids.front());
        const std::string tag = "marker#" + std::to_string(marker.size()) + " ";
        check(piece.special, tag + "is reported as special");
        check(piece.text == marker,
              tag + "carries its own spelling as vocabulary bytes",
              "piece bytes=" + std::to_string(piece.text.size()));
        check(engine.detokenize(ids, false) == marker, tag + "decodes to its own spelling");
        check(engine.detokenize(ids, true).empty(),
              tag + "skipping special tokens drops its spelling",
              "decoded bytes=" + std::to_string(engine.detokenize(ids, true).size()));
    }

    // An added token that is not a control token keeps its spelling in both views.
    const std::vector<ninfer::TokenId> think = engine.tokenize_text("<think>");
    check(think.size() == 1, "a named added token is a single id",
          "ids=" + std::to_string(think.size()));
    if (think.size() == 1) {
        const ninfer::TokenPiece piece = engine.token_piece(think.front());
        check(piece.text == "<think>" && !piece.special,
              "an added token that is not a control token keeps its spelling and its flag",
              "piece=" + piece.text);
        check(engine.detokenize(think, true) == "<think>",
              "suppression leaves a non-control added token alone");
    }

    // An id outside the checkpoint vocabulary is rejected by name, not silently decoded.
    bool threw = false;
    std::string detail;
    try {
        (void)engine.token_piece(1'000'000'000);
    } catch (const std::exception& error) {
        threw = true;
        detail = error.what();
    }
    check(threw && detail.find("outside the checkpoint vocabulary") != std::string::npos,
          "an out-of-vocabulary id is rejected", detail);

    return g_failures;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        return exercise(artifact) == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}