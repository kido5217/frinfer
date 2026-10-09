# Immutable loading and mutable execution have separate compile owners.
add_library(ninfer_model_loading STATIC registry.cpp)
ninfer_internal_includes(ninfer_model_loading)
target_link_libraries(ninfer_model_loading PUBLIC ninfer_artifact ninfer_text ninfer_ops)

# FrInfer-owned adapter over the vendored llama.cpp chat-parsing stack (third_party/llama-chat):
# the parse core behind the frontend's output session.
add_library(ninfer_chat_parsing STATIC qwen3_5/frontend/chat_parse_core.cpp)
ninfer_internal_includes(ninfer_chat_parsing)
target_link_libraries(ninfer_chat_parsing PUBLIC ninfer_llama_chat ninfer::json)

add_library(ninfer_model_runtime STATIC)
ninfer_internal_includes(ninfer_model_runtime)
# Constraint grammar composition (tool_grammar.cpp) goes through the vendor-neutral text-constraint
# adapter (text/model_grammar.h); the model runtime no longer compiles against the vendored
# XGrammar sources.
target_link_libraries(ninfer_model_runtime
  PUBLIC ninfer_model_loading ninfer_ops ninfer_runtime_support ninfer_text_grammar
  PRIVATE ninfer_media_decode ninfer_jinja ninfer_chat_parsing)


include(qwen3_5/loading_sources.cmake)
include(qwen3_5/frontend_sources.cmake)
include(qwen3_5/execution_sources.cmake)
include(qwen3_5/program_sources.cmake)
