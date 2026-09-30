#pragma once

// NInfer compat shim for llama.cpp's src/llama-sampler.h. llama-grammar.cpp includes it,
// but no symbol from it is used (verified at the vendored baseline by grep); the file
// exists so the vendored include does not need a patch.
