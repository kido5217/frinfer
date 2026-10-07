add_executable(frinfer
  cli/main.cpp
  cli/options.cpp)
ninfer_internal_includes(frinfer)
target_link_libraries(frinfer PRIVATE
  ninfer_engine
  ninfer_product_logging
  ninfer_product_prompt_input
  ninfer_model_acquire
  ninfer_build_info
  # The protocol-neutral constraint contract (JSON-Schema allowlist + conversion and the
  # fail-closed codes, src/product/constraint) is shared with the serve gateway.
  ninfer_constraint)

add_executable(frinfer-serve serve/main.cpp)
ninfer_internal_includes(frinfer-serve)
target_link_libraries(frinfer-serve PRIVATE ninfer_serve ninfer_model_acquire ninfer_product_logging ninfer_build_info)

add_executable(frinfer-perplexity
  perplexity/main.cpp
  perplexity/corpus.cpp
  perplexity/evaluation.cpp)
ninfer_internal_includes(frinfer-perplexity)
target_link_libraries(frinfer-perplexity PRIVATE
  ninfer_engine ninfer_product_logging ninfer_build_info ninfer::json)
