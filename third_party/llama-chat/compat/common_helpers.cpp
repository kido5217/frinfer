// NInfer compat: small self-contained helpers the vendored chat layer calls whose
// upstream definitions live in excluded translation units (common/common.cpp,
// common/chat-auto-parser-helpers.cpp). Bodies match upstream semantics.

#include "chat-auto-parser-helpers.h"
#include "common.h"

#include <cctype>
#include <sstream>
#include <string>
#include <vector>

// common/common.cpp
std::string string_join(const std::vector<std::string>& values, const std::string& separator) {
    std::ostringstream result;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) { result << separator; }
        result << values[i];
    }
    return result.str();
}

std::vector<std::string> string_split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end   = str.find(delimiter);
    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end   = str.find(delimiter, start);
    }
    parts.push_back(str.substr(start));
    return parts;
}

std::string string_repeat(const std::string& str, size_t n) {
    if (n == 0) { return ""; }
    std::string result;
    result.reserve(str.length() * n);
    for (size_t i = 0; i < n; ++i) { result += str; }
    return result;
}

void string_replace_all(std::string& s, const std::string& search, const std::string& replace) {
    if (search.empty()) { return; }
    std::string builder;
    builder.reserve(s.length());
    size_t pos      = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

// common/chat-auto-parser-helpers.cpp
std::string trim_whitespace(const std::string& str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }
    if (start == str.length()) { return ""; }
    size_t end = str.length() - 1;
    while (end > start && std::isspace(static_cast<unsigned char>(str[end]))) { end--; }
    return str.substr(start, end - start + 1);
}

std::string trim_leading_whitespace(const std::string& str) {
    size_t start = 0;
    while (start < str.length() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }
    return str.substr(start);
}

std::string trim_trailing_whitespace(const std::string& str) {
    if (str.empty()) { return ""; }
    size_t end = str.length() - 1;
    while (end > 0 && std::isspace(static_cast<unsigned char>(str[end]))) { end--; }
    if (end == 0 && std::isspace(static_cast<unsigned char>(str[0]))) { return ""; }
    return str.substr(0, end + 1);
}

std::string trim_trailing_newlines(const std::string& str) {
    size_t end = str.length();
    while (end > 0 && str[end - 1] == '\n') { end--; }
    return str.substr(0, end);
}
