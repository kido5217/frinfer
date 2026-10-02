# Immutable loading and mutable execution have separate compile owners.
add_library(ninfer_model_loading STATIC registry.cpp)
ninfer_internal_includes(ninfer_model_loading)
target_link_libraries(ninfer_model_loading PUBLIC ninfer_artifact ninfer_text ninfer_ops)

# NInfer-owned adapter over the vendored llama.cpp chat-parsing stack (third_party/llama-chat):
# the parse core behind the frontend's output session.
add_library(ninfer_chat_parsing STATIC qwen3_5/frontend/chat_parse_core.cpp)
ninfer_internal_includes(ninfer_chat_parsing)
target_link_libraries(ninfer_chat_parsing PUBLIC ninfer_llama_chat ninfer::json)

# Host-side GBNF mask module over the vendored llama.cpp grammar runtime
# (third_party/llama-grammar): compile, per-request state, the compiled token-trie mask
# producer (design #59, D2a compile-time placement) and the vendored engine walk retained
# as its differential oracle. See grammar/grammar.h.
add_library(ninfer_grammar STATIC qwen3_5/frontend/grammar/grammar.cpp
                                  qwen3_5/frontend/grammar/trie_producer.cpp
                                  qwen3_5/frontend/grammar/grammar_runtime.cpp
                                  qwen3_5/frontend/grammar/thinking_wrapper.cpp)
ninfer_internal_includes(ninfer_grammar)
target_link_libraries(ninfer_grammar PRIVATE ninfer_llama_grammar ninfer_llama_chat)

add_library(ninfer_model_runtime STATIC)
ninfer_internal_includes(ninfer_model_runtime)
target_link_libraries(ninfer_model_runtime
  PUBLIC ninfer_model_loading ninfer_ops ninfer_runtime_support ninfer_grammar
  PRIVATE ninfer_media_decode ninfer_jinja ninfer_chat_parsing)


include(qwen3_5/loading_sources.cmake)
include(qwen3_5/frontend_sources.cmake)
include(qwen3_5/execution_sources.cmake)
include(qwen3_5/program_sources.cmake)
