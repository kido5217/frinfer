// Tiny driver: JSON schema -> GBNF string, using the in-tree vendored converter
// (third_party/llama-chat/common/json-schema-to-grammar.cpp, byte-identical to
// llama.cpp at the vendored baseline). Prints the grammar to stdout.
//
// Build: tools/spike/mask-cost/build.sh <build-dir>

#include "json-schema-to-grammar.h"
#include "json.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <schema.json>\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    try {
        const common_json schema  = common_json::parse(buffer.str());
        const std::string grammar = json_schema_to_grammar(schema, /*force_gbnf=*/true);
        std::fputs(grammar.c_str(), stdout);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "conversion failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
