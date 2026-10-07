// End-to-end constrained generation matrix through the public Engine route (wayfinder map #45,
// ticket #53).
//
// The acceptance fixtures run against a real artifact under the ordinary and MTP backends: the
// g1-g3 grammars and the diary JSON schema from tests/fixtures/grammar/, plus multi-byte
// literals, a small schema with reachable maxLength/maxItems bounds, an empty root, a literal
// continuation whose drafts survive verify columns, and the invalid/unsatisfiable grammar
// contracts. Each constrained output is checked against its constraint independently of the
// mask producer: greedy sampling makes a run deterministic, but the model's chosen text inside
// the constraint is not golden.
//
// The checks mirror the engine's documented bounded-repetition semantics (a class repetition
// counts pieces, so a string bound can overrun by at most the longest piece; see the #59
// ruling); the MTP runs record draft acceptance per verify column.
//
// NINFER_TEST_ARTIFACT selects the artifact; without it the test skips (exit 77).

#include "ninfer/engine.h"

#include "json-schema-to-grammar.h"
#include "json.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

// The documented bounded-repetition overrun of the vendored engine (wayfinder #59): a class
// repetition counts pieces, so a string of maxLength N can reach N + the longest piece.
constexpr std::size_t kPieceOverrun = 127;

int g_failures = 0;

void check(bool ok, const std::string& what, const std::string& detail = {}) {
    if (ok) { return; }
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s%s%s\n", what.c_str(), detail.empty() ? "" : ": ",
                 detail.c_str());
}

void require_stop(const ninfer::GenerationResult& result, const std::string& where) {
    check(result.finish_reason == ninfer::FinishReason::StopToken,
          where + ": did not stop through the grammar's end",
          "finish_reason=" + std::to_string(static_cast<int>(result.finish_reason)));
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open fixture: " + path); }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::size_t utf8_length(const std::string& text) {
    std::size_t count = 0;
    for (const unsigned char byte : text) {
        if ((byte & 0xC0U) != 0x80U) { ++count; }
    }
    return count;
}

std::string preview(const std::string& text, std::size_t limit = 96) {
    std::string out;
    for (const char byte : text) {
        if (out.size() >= limit) {
            out += "...";
            break;
        }
        switch (byte) {
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += byte; break;
        }
    }
    return out;
}

std::string hex_preview(const std::string& text, std::size_t limit = 24) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    for (std::size_t index = 0; index < text.size() && index < limit; ++index) {
        const unsigned char byte = static_cast<unsigned char>(text[index]);
        out += ' ';
        out += digits[byte >> 4];
        out += digits[byte & 0xFU];
    }
    if (text.size() > limit) { out += " ..."; }
    return out;
}

std::string json_schema_grammar(const std::string& schema_text) {
    return json_schema_to_grammar(common_json::parse(schema_text), true);
}

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t end = text.find(separator, start);
        fields.push_back(text.substr(start, end == std::string::npos ? end : end - start));
        if (end == std::string::npos) { return fields; }
        start = end + 1;
    }
}

constexpr std::string_view kActs[] = {"none",     "gesture", "arm_task", "base",
                                      "approach", "power",   "view",     "delegate"};
constexpr std::string_view kPairKeys[] = {"presay",      "power",       "emote",
                                          "gesture",     "delegate",    "arm_task",
                                          "approach",    "base",        "base_amount",
                                          "needs_view",  "recall",      "look",
                                          "halt"};

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.rfind(prefix, 0) == 0;
}

template <std::size_t N>
bool in(std::string_view value, const std::string_view (&items)[N]) {
    for (const std::string_view item : items) {
        if (value == item) { return true; }
    }
    return false;
}

bool complete_act(std::string_view field) {
    return starts_with(field, "act=") && in(field.substr(4), kActs);
}

// Anything the stream may still grow into "act=<one of the acts>".
bool act_prefix(std::string_view field) {
    for (const std::string_view act : kActs) {
        const std::string full = "act=" + std::string(act);
        if (field.size() <= full.size() && starts_with(full, field)) { return true; }
    }
    return false;
}

bool complete_pair(std::string_view field) {
    const std::size_t equal = field.find('=');
    return equal != std::string_view::npos && in(field.substr(0, equal), kPairKeys);
}

// A prefix of "key" or "key=<any value>": what the stream may still grow into a pair.
bool pair_prefix(std::string_view field) {
    for (const std::string_view key : kPairKeys) {
        if (field.size() <= key.size() && starts_with(key, field)) { return true; }
        if (field.size() > key.size() && starts_with(field, key) && field[key.size()] == '=') {
            return true;
        }
    }
    return false;
}

bool complete_say(std::string_view field) {
    return starts_with(field, "say=") && field.size() > 4;
}

bool say_prefix(std::string_view field) {
    constexpr std::string_view tag = "say=";
    if (field.size() < tag.size()) { return starts_with(tag, field); }
    return starts_with(field, tag);
}

bool tone_prefix(std::string_view field) {
    constexpr std::string_view tag = "tone=";
    if (field.size() < tag.size()) { return starts_with(tag, field); }
    return starts_with(field, tag);
}

// Independent mirror of tests/fixtures/grammar/g3_key_value_record.gbnf.
bool record_conforms(const std::string& text, std::string& why) {
    if (text.find('\n') != std::string::npos) { why = "contains a newline"; return false; }
    const std::vector<std::string> fields = split(text, ';');
    if (fields.size() < 3) { why = "fewer than three ';'-separated fields"; return false; }
    if (!complete_act(fields.front())) {
        why = "does not start with act=<one of the acts>";
        return false;
    }
    const std::string& say = fields[fields.size() - 2];
    const std::string& tone = fields[fields.size() - 1];
    if (!complete_say(say)) { why = "missing a non-empty say="; return false; }
    if (!starts_with(tone, "tone=")) { why = "missing tone="; return false; }
    for (std::size_t index = 1; index + 2 < fields.size(); ++index) {
        const std::string& field = fields[index];
        if (!complete_pair(field)) {
            why = "pair key outside the grammar's set: " + field;
            return false;
        }
    }
    return true;
}

// A record stream cut by the token limit: every finished field is complete and the last may
// still be growing into its construct.
bool record_prefix_conforms(const std::string& text, std::string& why) {
    if (text.find('\n') != std::string::npos) { why = "contains a newline"; return false; }
    const std::vector<std::string> fields = split(text, ';');
    if (!complete_act(fields.front()) && !act_prefix(fields.front())) {
        why = "does not start with act=";
        return false;
    }
    if (fields.size() > 1 && !complete_act(fields.front())) {
        why = "the act field is incomplete mid-record";
        return false;
    }
    bool seen_say = false;
    for (std::size_t index = 1; index + 1 < fields.size(); ++index) {
        const std::string& field = fields[index];
        if (!seen_say) {
            if (complete_pair(field)) { continue; }
            if (complete_say(field)) {
                seen_say = true;
                continue;
            }
            why = "field is neither a pair nor say=: " + field;
            return false;
        }
        why = "field after say=: " + field;
        return false;
    }
    if (fields.size() == 1) { return true; }
    const std::string& last = fields.back();
    if (!seen_say) {
        if (complete_pair(last) || pair_prefix(last)) { return true; }
        if (complete_say(last) || say_prefix(last)) { return true; }
        why = "the truncated field is neither a pair nor a say= prefix: " + last;
        return false;
    }
    if (tone_prefix(last)) { return true; }
    why = "the truncated field is not a tone= prefix: " + last;
    return false;
}

// tests/fixtures/grammar/diary.json independently: exact member set, types and bounds. String
// bounds allow the documented piece overrun.
bool diary_conforms(const std::string& text, std::string& why) {
    common_json document;
    try {
        document = common_json::parse(text);
    } catch (const common_json_error& error) {
        why = std::string("not JSON: ") + error.what();
        return false;
    }
    if (!document.is_object()) { why = "root is not an object"; return false; }
    if (document.size() != 4) { why = "member count is not exactly 4"; return false; }
    for (const char* key : {"title", "text", "facts", "people"}) {
        if (!document.contains(key)) { why = std::string("missing ") + key; return false; }
    }
    const auto string_member = [&](const char* key, std::size_t max_length) {
        const common_json& value = document.at(key);
        if (!value.is_string()) { why = std::string(key) + " is not a string"; return false; }
        if (utf8_length(value.get<std::string>()) > max_length + kPieceOverrun) {
            why = std::string(key) + " exceeds its maximum length";
            return false;
        }
        return true;
    };
    if (!string_member("title", 60)) { return false; }
    if (!string_member("text", 2400)) { return false; }
    const common_json& facts = document.at("facts");
    if (!facts.is_array() || facts.size() > 14) {
        why = "facts is not an array of at most 14";
        return false;
    }
    for (std::size_t index = 0; index < facts.size(); ++index) {
        const common_json& item = facts.at(index);
        if (!item.is_string() || utf8_length(item.get<std::string>()) > 120 + kPieceOverrun) {
            why = "a facts item is not a bounded string";
            return false;
        }
    }
    const common_json& people = document.at("people");
    if (!people.is_array() || people.size() > 4) {
        why = "people is not an array of at most 4";
        return false;
    }
    for (std::size_t index = 0; index < people.size(); ++index) {
        const common_json& person = people.at(index);
        if (!person.is_object() || person.size() != 2 || !person.contains("name") ||
            !person.contains("when")) {
            why = "a people item is not {name, when}";
            return false;
        }
        if (!person.at("name").is_string() ||
            utf8_length(person.at("name").get<std::string>()) > 30 + kPieceOverrun) {
            why = "a people name exceeds its maximum length";
            return false;
        }
        if (!person.at("when").is_string() ||
            utf8_length(person.at("when").get<std::string>()) > 5 + kPieceOverrun) {
            why = "a people when exceeds its maximum length";
            return false;
        }
    }
    return true;
}

// The reachable-bound schema: exactly one member, an array of at most two short strings.
bool small_schema_conforms(const std::string& text, std::string& why) {
    common_json document;
    try {
        document = common_json::parse(text);
    } catch (const common_json_error& error) {
        why = std::string("not JSON: ") + error.what();
        return false;
    }
    if (!document.is_object() || document.size() != 1 || !document.contains("a")) {
        why = "root is not an object holding only \"a\"";
        return false;
    }
    const common_json& array = document.at("a");
    if (!array.is_array() || array.size() > 2) {
        why = "a is not an array of at most 2";
        return false;
    }
    for (std::size_t index = 0; index < array.size(); ++index) {
        const common_json& item = array.at(index);
        if (!item.is_string() || utf8_length(item.get<std::string>()) > 3 + kPieceOverrun) {
            why = "an a item is not a bounded string";
            return false;
        }
    }
    return true;
}

struct Case {
    std::string name;
    std::string grammar;
    std::uint32_t max_tokens = 0;
    std::function<void(const ninfer::GenerationResult&, const std::string&)> verify;
};

ninfer::RequestOptions greedy_request(std::uint32_t max_tokens, const std::string& grammar) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = max_tokens;
    request.execution.sampling.temperature    = 0.0F;
    if (!grammar.empty()) {
        request.constraint.emplace(ninfer::GrammarConstraint{grammar});
    }
    return request;
}

ninfer::EngineOptions engine_options(const char* artifact, ninfer::SpeculativeBackend backend) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 4096;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                        = 1024;
    options.speculative.backend                  = backend;
    if (backend == ninfer::SpeculativeBackend::Mtp) {
        options.speculative.draft_tokens = 3;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_capacity_bytes    = 1ULL << 30;
    return options;
}

void report(const std::string& where, const ninfer::GenerationResult& result) {
    std::printf("matrix %-28s finish=%d tokens=%zu content=\"%s\"\n", where.c_str(),
                static_cast<int>(result.finish_reason), result.generated_token_ids.size(),
                preview(result.content).c_str());
    if (result.speculative.enabled) {
        std::printf("    mtp rounds=%llu drafted=%llu accepted=%llu per_position=[",
                    static_cast<unsigned long long>(result.speculative.rounds),
                    static_cast<unsigned long long>(result.speculative.drafted_tokens),
                    static_cast<unsigned long long>(result.speculative.accepted_tokens));
        for (std::size_t index = 0; index < result.speculative.accepted_per_position.size();
             ++index) {
            std::printf("%s%llu", index == 0 ? "" : ", ",
                        static_cast<unsigned long long>(
                            result.speculative.accepted_per_position[index]));
        }
        std::printf("]\n");
    }
}

void verify_g1(const ninfer::GenerationResult& result, const std::string& where) {
    require_stop(result, where);
    check(result.content == "да" || result.content == "нет",
          where + ": output is outside the alternation", preview(result.content));
}

void verify_g2(const ninfer::GenerationResult& result, const std::string& where) {
    require_stop(result, where);
    check(result.content.find('\n') == std::string::npos,
          where + ": output contains a newline", preview(result.content));
    const std::size_t length = utf8_length(result.content);
    check(length >= 3 && length <= 240 + kPieceOverrun,
          where + ": output length is outside the bounded line", std::to_string(length));
}

void verify_g3(const ninfer::GenerationResult& result, const std::string& where) {
    std::string why;
    if (result.finish_reason == ninfer::FinishReason::StopToken) {
        check(record_conforms(result.content, why), where + ": record does not conform",
              why + " | " + preview(result.content));
        return;
    }
    if (result.finish_reason == ninfer::FinishReason::OutputLimit) {
        // The fixture's `rest ::= (";" pair)*` is unbounded and greedy decoding loops
        // `;base=act` pairs, so the run is expected to end at the token limit; the emitted
        // stream must still be a legal prefix of the record grammar.
        check(record_prefix_conforms(result.content, why),
              where + ": truncated record is not a legal prefix",
              why + " | " + preview(result.content));
        std::printf("    note: the unbounded pair loop ended at the token limit\n");
        return;
    }
    check(false, where + ": unexpected finish reason",
          "finish_reason=" + std::to_string(static_cast<int>(result.finish_reason)));
}

void verify_diary(const ninfer::GenerationResult& result, const std::string& where) {
    require_stop(result, where);
    std::string why;
    if (!diary_conforms(result.content, why)) {
        check(false, where + ": diary does not conform", why + " | " + preview(result.content));
        return;
    }
    const common_json document = common_json::parse(result.content);
    std::printf("    diary title=%zu text=%zu facts=%zu people=%zu\n",
                utf8_length(document.at("title").get<std::string>()),
                utf8_length(document.at("text").get<std::string>()),
                document.at("facts").size(), document.at("people").size());
}

void verify_small_schema(const ninfer::GenerationResult& result, const std::string& where) {
    require_stop(result, where);
    std::string why;
    check(small_schema_conforms(result.content, why), where + ": schema output does not conform",
          why + " | " + preview(result.content));
}

void verify_cjk(const ninfer::GenerationResult& result, const std::string& where) {
    require_stop(result, where);
    check(result.content == "你好，世界", where + ": the literal grammar did not force the text",
          preview(result.content));
}

void verify_emoji(const ninfer::GenerationResult& result, const std::string& where) {
    require_stop(result, where);
    check(result.content == "🎉🚀", where + ": the literal grammar did not force the text",
          preview(result.content));
}

void verify_empty_root(const ninfer::GenerationResult& result, const std::string& where) {
    require_stop(result, where);
    check(result.content.empty(), where + ": the empty root produced content",
          preview(result.content));
}

void verify_fib(const ninfer::GenerationResult& result, const std::string& where) {
    require_stop(result, where);
    check(result.content == "1, 1, 2, 3, 5",
          where + ": the literal grammar did not force the text", preview(result.content));
}

void run_cases(ninfer::Engine& engine, const std::string& backend, bool mtp,
               const std::vector<Case>& cases) {
    for (const Case& test_case : cases) {
        const std::string where = backend + "/" + test_case.name;
        try {
            const ninfer::GenerationResult result = engine.generate(
                engine.prepare_tokens(engine.tokenize_text("Complete the sequence: ")),
                greedy_request(test_case.max_tokens, test_case.grammar));
            report(where, result);
            test_case.verify(result, where);
            if (mtp && test_case.name == "fib-literal") {
                // The model's own greedy continuation is the literal, so drafts survive the
                // masked verify columns; pin that the multi-column path was actually exercised.
                check(result.speculative.accepted_tokens != 0, where + ": no draft was accepted");
                bool multi_column = false;
                for (std::size_t index = 1;
                     index < result.speculative.accepted_per_position.size(); ++index) {
                    if (result.speculative.accepted_per_position[index] != 0) {
                        multi_column = true;
                    }
                }
                check(multi_column,
                      where + ": no draft past the first verify column was accepted (the masked "
                              "multi-column path was not exercised)");
            }
        } catch (const std::exception& error) {
            check(false, where + ": generation threw", error.what());
        }
    }
}

// A grammar with an empty language can never be satisfied. The contract is fail-closed: the
// request is rejected at submit or generation fails, and no content is emitted through it.
void expect_unsatisfiable_fail_closed(ninfer::Engine& engine, const std::string& where,
                                      const std::string& grammar) {
    try {
        const ninfer::GenerationResult result = engine.generate(
            engine.prepare_tokens(engine.tokenize_text("Complete the sequence: ")),
            greedy_request(32, grammar));
        if (result.content.empty() && result.finish_reason == ninfer::FinishReason::StopToken) {
            std::printf("matrix %-28s stopped empty (no token emitted)\n", where.c_str());
            return;
        }
        check(false, where + ": an unsatisfiable grammar produced content",
              preview(result.content) + " bytes:" + hex_preview(result.content));
    } catch (const ninfer::RequestError& error) {
        std::printf("matrix %-28s rejected: %s\n", where.c_str(), error.what());
    } catch (const std::logic_error& error) {
        // The fail-closed path for a state the mask cannot advance: the engine's divergence
        // check, not an unrelated failure.
        const std::string what = error.what();
        check(what.find("diverged from its grammar") != std::string::npos,
              where + ": generation failed with an unexpected logic error", what);
        std::printf("matrix %-28s failed closed during generation: %s\n", where.c_str(),
                    error.what());
    } catch (const std::exception& error) {
        check(false, where + ": generation failed with an unexpected error", error.what());
    }
}

void expect_invalid_grammar(ninfer::Engine& engine, const std::string& where,
                            const std::string& grammar) {
    try {
        (void)engine.generate(engine.prepare_tokens(engine.tokenize_text("Complete the sequence: ")),
                              greedy_request(8, grammar));
        check(false, where + ": an invalid grammar was not rejected");
    } catch (const ninfer::RequestError& error) {
        check(error.kind() == ninfer::RequestErrorKind::InvalidConstraint,
              where + ": rejected with an unexpected kind");
        std::printf("matrix %-28s rejected: %s\n", where.c_str(), error.what());
    }
}

int exercise(const char* artifact) {
    const std::string fixtures = std::string(NINFER_SOURCE_DIR) + "/tests/fixtures/grammar/";
    const std::string g1         = read_file(fixtures + "g1_literal_alternation.gbnf");
    const std::string g2         = read_file(fixtures + "g2_bounded_line.gbnf");
    const std::string g3         = read_file(fixtures + "g3_key_value_record.gbnf");
    const std::string diary      = json_schema_grammar(read_file(fixtures + "diary.json"));
    const std::string small = json_schema_grammar(
        R"schema({"type":"object","additionalProperties":false,"required":["a"],"properties":{"a":{"type":"array","maxItems":2,"items":{"type":"string","maxLength":3}}}})schema");

    const std::vector<Case> cases = {
        {"g1-alternation", g1, 16, verify_g1},
        {"g2-bounded-line", g2, 384, verify_g2},
        {"g3-key-value", g3, 512, verify_g3},
        {"diary-schema", diary, 2048, verify_diary},
        {"small-schema", small, 128, verify_small_schema},
        {"cjk-literal", "root ::= \"你好，世界\"", 32, verify_cjk},
        {"emoji-literal", "root ::= \"🎉🚀\"", 32, verify_emoji},
        {"empty-root", "root ::= \"\"", 8, verify_empty_root},
        {"fib-literal", "root ::= \"1, 1, 2, 3, 5\"", 16, verify_fib},
    };

    {
        ninfer::Engine engine(engine_options(artifact, ninfer::SpeculativeBackend::None));
        run_cases(engine, "none", false, cases);
        expect_unsatisfiable_fail_closed(engine, "none/unsatisfiable", "root ::= \"\\x00\"");
        expect_invalid_grammar(engine, "none/invalid-syntax", "root ::= [");
    }
    {
        ninfer::Engine engine(engine_options(artifact, ninfer::SpeculativeBackend::Mtp));
        run_cases(engine, "mtp", true, cases);
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "constrained matrix: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("ok constrained matrix\n");
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        return exercise(artifact);
    } catch (const std::exception& error) {
        std::cerr << "constrained matrix failed: " << error.what() << '\n';
        return 1;
    }
}
