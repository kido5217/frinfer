#pragma once

// NInfer compat shim for llama.cpp's src/llama-impl.h, reduced to the logging and
// assertion macros llama-grammar.cpp uses. The macro bodies mirror
// third_party/llama-chat/compat/ggml.h, which the in-tree consumer of vendored llama.cpp
// sources already uses.

#include <cstdio>
#include <cstdlib>

#define LLAMA_LOG_ERROR(...)                                                                       \
    do {                                                                                           \
        std::fprintf(stderr, "llama-grammar error: " __VA_ARGS__);                                 \
        std::fflush(stderr);                                                                       \
    } while (0)

#define LLAMA_LOG_WARN(...)                                                                        \
    do {                                                                                           \
        std::fprintf(stderr, "llama-grammar warning: " __VA_ARGS__);                               \
        std::fflush(stderr);                                                                       \
    } while (0)

#define LLAMA_LOG_DEBUG(...) ((void)0)

#define GGML_ASSERT(expr)                                                                          \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            std::fprintf(stderr, "llama-grammar: GGML_ASSERT(%s) failed (%s:%d)\n", #expr,         \
                         __FILE__, __LINE__);                                                      \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

#define GGML_ABORT(...)                                                                            \
    do {                                                                                           \
        std::fprintf(stderr, "llama-grammar: GGML_ABORT (%s:%d): ", __FILE__, __LINE__);           \
        std::fprintf(stderr, __VA_ARGS__);                                                         \
        std::fprintf(stderr, "\n");                                                                \
        std::abort();                                                                              \
    } while (0)
