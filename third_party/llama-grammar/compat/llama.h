#pragma once

// NInfer compat shim for llama.cpp's include/llama.h, reduced to what the vendored
// llama-grammar.{h,cpp} references: the llama_token typedef and the candidate-array types
// the grammar filter writes into. Shapes mirror llama.cpp at the vendored baseline
// (include/llama.h:69, 231-245, 05af0d2b1398394cfa67e1918fee7feabccaa9bc).

#include <cstddef>
#include <cstdint>

typedef int32_t llama_token;

typedef struct llama_token_data {
    llama_token id; // token id
    float logit;    // log-odds of the token
    float p;        // probability of the token
} llama_token_data;

typedef struct llama_token_data_array {
    // NOTE: this pointer can be modified by the samplers
    llama_token_data* data;
    size_t size;
    int64_t selected; // this is the index in the data array (i.e. not the token id)
    bool sorted;      // note: do not assume the data is sorted - always check this flag
} llama_token_data_array;
