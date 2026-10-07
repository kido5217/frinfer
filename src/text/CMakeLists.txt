add_library(ninfer_text STATIC
  unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c
)
ninfer_internal_includes(ninfer_text)
target_include_directories(ninfer_text PRIVATE ${PROJECT_SOURCE_DIR}/third_party)

# Language evaluation reuses the engine's Unicode and JSON dependencies.
target_sources(ninfer_jinja PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/jinja.cpp)
ninfer_internal_includes(ninfer_jinja)
target_include_directories(ninfer_jinja PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer_jinja PRIVATE ninfer_text ninfer::json)

# Upstream XGrammar-backed text constraint core (ticket #245): GBNF, JSON object/schema, choice
# and regex compilation on the request-owned GrammarSession/GrammarCompiler, replacing the fork's
# in-tree trie stack. Named ninfer_text_grammar while the fork's models-level ninfer_grammar target
# still exists; the name collapses to upstream's ninfer_grammar when the fork stack is retired.
add_library(ninfer_text_grammar STATIC
  grammar.cpp
  json_schema.cpp
  schema_composition.cpp
  json_input.cpp)
ninfer_internal_includes(ninfer_text_grammar)
target_include_directories(ninfer_text_grammar PRIVATE ${PROJECT_SOURCE_DIR}/third_party/xgrammar/cpp)
target_include_directories(ninfer_text_grammar SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson)
target_link_libraries(ninfer_text_grammar PRIVATE ninfer_xgrammar ninfer_text ninfer::json)
