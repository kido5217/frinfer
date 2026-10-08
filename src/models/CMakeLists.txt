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
# The frontend's constraint grammar builders (tool_grammar.cpp, ticket #242) compile against the
# vendored XGrammar C++ sources directly, so the model runtime needs the same private include
# path ninfer_text_grammar uses.
target_include_directories(ninfer_model_runtime PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/cpp)
target_include_directories(ninfer_model_runtime SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson)
target_link_libraries(ninfer_model_runtime
  PUBLIC ninfer_model_loading ninfer_ops ninfer_runtime_support ninfer_text_grammar
  PRIVATE ninfer_media_decode ninfer_jinja ninfer_chat_parsing ninfer_xgrammar)


include(qwen3_5/loading_sources.cmake)
include(qwen3_5/frontend_sources.cmake)
include(qwen3_5/execution_sources.cmake)
include(qwen3_5/program_sources.cmake)
