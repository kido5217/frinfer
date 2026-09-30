#pragma once
// Stand-in for llama.cpp's llama-impl.h for the standalone grammar build:
// LLAMA_LOG_* go to stderr, GGML_ASSERT/GGML_ABORT abort. Mirrors the macro
// bodies of third_party/llama-chat/compat/ggml.h, which the in-tree consumer
// of vendored llama.cpp sources already uses.

#include <cstdio>
#include <cstdlib>

#define LLAMA_LOG_ERROR(...)                                                                       \
    do {                                                                                           \
        std::fprintf(stderr, __VA_ARGS__);                                                         \
        std::fflush(stderr);                                                                       \
    } while (0)

#define LLAMA_LOG_WARN(...)                                                                        \
    do {                                                                                           \
        std::fprintf(stderr, __VA_ARGS__);                                                         \
        std::fflush(stderr);                                                                       \
    } while (0)

#define LLAMA_LOG_DEBUG(...)                                                                       \
    do {                                                                                           \
        if (std::getenv("MASK_COST_VERBOSE") != nullptr) {                                         \
            std::fprintf(stderr, __VA_ARGS__);                                                     \
            std::fflush(stderr);                                                                   \
        }                                                                                          \
    } while (0)

#define GGML_ASSERT(expr)                                                                          \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            std::fprintf(stderr, "mask-cost: GGML_ASSERT(%s) failed (%s:%d)\n", #expr, __FILE__,   \
                         __LINE__);                                                                \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

#define GGML_ABORT(...)                                                                            \
    do {                                                                                           \
        std::fprintf(stderr, "mask-cost: GGML_ABORT (%s:%d): ", __FILE__, __LINE__);               \
        std::fprintf(stderr, __VA_ARGS__);                                                         \
        std::fprintf(stderr, "\n");                                                                \
        std::abort();                                                                              \
    } while (0)
