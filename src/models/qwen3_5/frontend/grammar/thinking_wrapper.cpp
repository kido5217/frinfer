#include "models/qwen3_5/frontend/grammar/thinking_wrapper.h"

#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

// The vendored parser's name alphabet (llama-grammar.cpp is_word_char), so identifier scanning
// agrees with the parser on where a rule name begins and ends.
bool is_word_char(char byte) {
    return (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
           (byte >= '0' && byte <= '9') || byte == '-';
}

std::string hex_byte(char byte) {
    char buffer[5];
    std::snprintf(buffer, sizeof(buffer), "\\x%02X", static_cast<unsigned char>(byte));
    return buffer;
}

// One marker byte in a GBNF string literal.
std::string literal_byte(char byte) {
    switch (byte) {
    case '"':
        return "\\\"";
    case '\\':
        return "\\\\";
    case '\n':
        return "\\n";
    case '\r':
        return "\\r";
    case '\t':
        return "\\t";
    default:
        break;
    }
    const auto value = static_cast<unsigned char>(byte);
    if (value < 0x20U || value == 0x7FU) { return hex_byte(byte); }
    return std::string(1, byte);
}

std::string literal(std::string_view text) {
    std::string out = "\"";
    for (const char byte : text) { out += literal_byte(byte); }
    out += "\"";
    return out;
}

// Negated single-byte class: `[^ab]` for excluded bytes a, b.
std::string excluded_class(std::string_view bytes) {
    std::string out = "[^";
    for (const char byte : bytes) {
        switch (byte) {
        case '\\':
            out += "\\\\";
            break;
        case ']':
            out += "\\]";
            break;
        case '^':
            out += "\\^";
            break;
        case '-':
            out += "\\-";
            break;
        default: {
            const auto value = static_cast<unsigned char>(byte);
            out += (value < 0x20U || value == 0x7FU) ? hex_byte(byte) : std::string(1, byte);
        }
        }
    }
    out += "]";
    return out;
}

// True when `name` is declared as a rule (`name` followed by optional whitespace and `::=`) in
// `gbnf`. Rule names are word-char runs, so a longer identifier never matches.
bool defines_rule(std::string_view gbnf, std::string_view name) {
    for (std::size_t found = gbnf.find(name); found != std::string_view::npos;
         found             = gbnf.find(name, found + 1)) {
        const std::size_t after = found + name.size();
        std::size_t cursor      = after;
        while (cursor < gbnf.size() && (gbnf[cursor] == ' ' || gbnf[cursor] == '\t' ||
                                        gbnf[cursor] == '\r' || gbnf[cursor] == '\n')) {
            ++cursor;
        }
        if (gbnf.substr(cursor, 3) == "::=") { return true; }
    }
    return false;
}

// Renames every `root` identifier - declaration and references alike, outside string literals,
// character classes and comments - to `answer_name`, preserving the input language. Returns false
// when the input declares no `root` rule.
bool rename_root(std::string_view gbnf, const std::string& answer_name, std::string* out) {
    out->clear();
    out->reserve(gbnf.size());
    std::size_t index = 0;
    while (index < gbnf.size()) {
        const char byte = gbnf[index];
        if (byte == '"' || byte == '[') {
            // Copy a string literal / character class verbatim, honoring backslash escapes.
            const char terminator = byte;
            out->push_back(byte);
            ++index;
            while (index < gbnf.size()) {
                if (gbnf[index] == '\\' && index + 1 < gbnf.size()) {
                    out->push_back(gbnf[index]);
                    out->push_back(gbnf[index + 1]);
                    index += 2;
                    continue;
                }
                const bool done = gbnf[index] == (terminator == '"' ? '"' : ']');
                out->push_back(gbnf[index]);
                ++index;
                if (done) { break; }
            }
            continue;
        }
        if (byte == '#') {
            while (index < gbnf.size() && gbnf[index] != '\n') { out->push_back(gbnf[index++]); }
            continue;
        }
        if (byte == '<') {
            // A token reference: `<[id]>` or `<text>` (llama-grammar parse_token). It is an
            // operand whose inner text is a token name, never a rule name, so the whole span is
            // copied verbatim.
            const std::size_t end  = gbnf.find('>', index + 1);
            const std::size_t stop = end == std::string_view::npos ? gbnf.size() : end + 1;
            out->append(gbnf.substr(index, stop - index));
            index = stop;
            continue;
        }
        if (is_word_char(byte)) {
            const std::size_t begin = index;
            while (index < gbnf.size() && is_word_char(gbnf[index])) { ++index; }
            const std::string_view name = gbnf.substr(begin, index - begin);
            if (name == "root") {
                *out += answer_name;
            } else {
                out->append(name);
            }
            continue;
        }
        out->push_back(byte);
        ++index;
    }
    return defines_rule(*out, answer_name);
}

// A rule-name prefix that does not occur anywhere in `gbnf`: the parser silently overwrites a
// rule on a duplicate definition, so absence of the prefix makes every wrapper name unique.
std::optional<std::string> unique_prefix(std::string_view gbnf) {
    for (int attempt = 0; attempt < 1000; ++attempt) {
        std::string prefix = "tcw" + std::to_string(attempt) + "-";
        if (gbnf.find(prefix) == std::string_view::npos) { return prefix; }
    }
    return std::nullopt;
}

} // namespace

std::optional<std::string> wrap_thinking_constraint_grammar(std::string_view gbnf,
                                                            std::string_view close_marker,
                                                            std::string* error) {
    if (error != nullptr) { error->clear(); }
    if (close_marker.empty()) {
        if (error != nullptr) { *error = "thinking wrapper needs a non-empty close marker"; }
        return std::nullopt;
    }
    const std::optional<std::string> prefix = unique_prefix(gbnf);
    if (!prefix) {
        if (error != nullptr) {
            *error = "grammar leaves no collision-free rule-name prefix for the thinking wrapper";
        }
        return std::nullopt;
    }

    const std::string answer = *prefix + "answer";
    std::string out;
    out.reserve(gbnf.size() + 256);
    if (!rename_root(gbnf, answer, &out)) {
        if (error != nullptr) { *error = "grammar does not define the 'root' rule"; }
        return std::nullopt;
    }
    if (!out.empty() && out.back() != '\n') { out += '\n'; }

    const std::string body   = *prefix + "body";
    const std::string close  = *prefix + "close";
    const std::string ws     = *prefix + "ws";
    const std::string marker = std::string(close_marker);
    const auto state = [&](std::size_t index) { return *prefix + "b" + std::to_string(index); };
    const std::string first = marker.substr(0, 1);

    // The chain over the close marker's prefixes. State k (1 <= k <= n-1) is the pending position
    // after the marker's first k bytes; a byte that neither continues the prefix nor repeats the
    // marker's first byte completes the state and returns to the body loop, and the first byte
    // restarts the chain. The last state has no transition on the marker's final byte: only the
    // root-level close consumes it, which is what forces the close once the full prefix is seen.
    out += "root ::= " + body + " " + close + " " + ws + " " + answer + " | " + body + "\n";
    out += body + " ::= " + state(0) + "*\n";
    if (marker.size() == 1) {
        out += state(0) + " ::= " + excluded_class(first) + "\n";
    } else {
        out += state(0) + " ::= " + excluded_class(first) + " | " + literal(first) + " " +
               state(1) + "\n";
        for (std::size_t index = 1; index < marker.size(); ++index) {
            const std::string name     = state(index);
            const std::string current  = marker.substr(index, 1);
            const std::string excluded = current + first;
            if (index + 1 == marker.size()) {
                out += name + " ::= " + excluded_class(excluded) + " | " + literal(first) + " " +
                       state(1) + "\n";
            } else if (index == 1) {
                out += name + " ::= " + literal(first) + "* ( " + literal(current) + " " +
                       state(2) + " | " + excluded_class(excluded) + " )\n";
            } else {
                out += name + " ::= " + excluded_class(excluded) + " | " + literal(current) + " " +
                       state(index + 1) + " | " + literal(first) + " " + state(1) + "\n";
            }
        }
    }
    out += close + " ::= " + literal(marker) + "\n";
    out += ws + " ::= [ \\t\\r\\n]+\n";
    return out;
}

} // namespace ninfer::models::qwen3_5::frontend
