add_library(ninfer_media_acquire STATIC
  media_acquire/acquire.cpp
)
ninfer_internal_includes(ninfer_media_acquire)
target_link_libraries(ninfer_media_acquire PRIVATE PkgConfig::LIBCURL)

add_library(ninfer_product_prompt_input STATIC
  prompt_input/prompt_input.cpp
)
ninfer_internal_includes(ninfer_product_prompt_input)
target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire ninfer::json)

# Protocol-neutral JSON Schema constrained-decoding contract, shared by the CLI and the HTTP
# serving gateway. It owns no protocol or transport dependency; the vendored llama.cpp converter
# is linked privately.
add_library(ninfer_constraint STATIC
  constraint/constraint_contract.cpp
)
ninfer_internal_includes(ninfer_constraint)
target_link_libraries(ninfer_constraint PUBLIC ninfer::json PRIVATE ninfer_llama_chat)

add_library(ninfer_model_acquire STATIC
  model_acquire/model_acquire.cpp
)
ninfer_internal_includes(ninfer_model_acquire)
# libcurl arrives with the streaming download slice; the slice-0 cache layout
# itself needs no transport.

add_library(ninfer_product_logging STATIC
  logging/logging.cpp
  logging/pretty_format.cpp
  logging/startup_log.cpp
)
ninfer_internal_includes(ninfer_product_logging)
target_link_libraries(ninfer_product_logging PUBLIC spdlog::spdlog)
