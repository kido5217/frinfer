# Vendored llama.cpp GBNF runtime; provenance, patch set and adaptation map:
# README.ninfer.md. The sources are carried byte-identical to the baseline except for the
# documented patch set in patches/.
add_library(ninfer_llama_grammar STATIC
  llama-grammar.cpp)

# compat/ carries the NInfer-owned shims (llama.h, llama-impl.h, llama-vocab.h,
# llama-sampler.h); the vendored include root stays first so a later vendored file of the
# same name resolves to the vendored copy.
target_include_directories(ninfer_llama_grammar
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/compat)
