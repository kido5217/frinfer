#pragma once

// FrInfer compat shim for llama.cpp's common/log.h.
// The vendored sources log through these macros; DBG/INF compile out, WRN/ERR go to
// stderr with a prefix. FrInfer's own logging does not consume these messages.

#include <cstdio>

#define LOG_DBG(...) ((void)0)
#define LOG_INF(...) ((void)0)
#define LOG_WRN(...) ((void)std::fprintf(stderr, "llama-chat warning: " __VA_ARGS__))
#define LOG_ERR(...) ((void)std::fprintf(stderr, "llama-chat error: " __VA_ARGS__))
