# Vendored llama.cpp chat-parsing stack; provenance and adaptation map: README.frinfer.md.
add_library(ninfer_llama_chat STATIC
  common/chat.cpp
  common/chat-peg-parser.cpp
  common/json.cpp
  common/json-schema.cpp
  common/json-schema-to-grammar.cpp
  common/peg-parser.cpp
  common/parsers/parsers.cpp
  common/parsers/qwen3-coder.cpp
  common/trie.cpp
  common/unicode.cpp
  compat/common_helpers.cpp
  compat/compat.cpp
  compat/excluded_stubs.cpp)

# compat/ carries the FrInfer-owned shims (common.h, log.h, ggml.h, jinja/caps.h,
# jinja/value.h); common/ is the verbatim upstream include root and must come first
# so vendored includes resolve to vendored files.
target_include_directories(ninfer_llama_chat
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/compat)

# chat.h exposes jinja types and common_json in its public API.
target_link_libraries(ninfer_llama_chat
  PUBLIC
    ninfer_jinja
    ninfer::json)
