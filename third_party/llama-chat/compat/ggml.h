#pragma once

// FrInfer compat shim for ggml's ggml.h, reduced to the two assertion macros the
// vendored chat sources use. There is no ggml dependency in this subtree.

#include <cstdio>
#include <cstdlib>

#define GGML_ASSERT(expr)                                                                          \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            std::fprintf(stderr, "llama-chat: GGML_ASSERT(%s) failed (%s:%d)\n", #expr, __FILE__,  \
                         __LINE__);                                                                \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

#define GGML_ABORT(...)                                                                            \
    do {                                                                                           \
        std::fprintf(stderr, "llama-chat: GGML_ABORT (%s:%d): ", __FILE__, __LINE__);              \
        std::fprintf(stderr, __VA_ARGS__);                                                         \
        std::fprintf(stderr, "\n");                                                                \
        std::abort();                                                                              \
    } while (0)
