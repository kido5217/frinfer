#pragma once

#include "ninfer/build_info.h"

#include <string>
#include <string_view>

namespace ninfer::product {

// The one-line identity printed by the product binaries' `--version`:
// "<program> <version> (<git sha>; <build config>)".
inline std::string version_text(std::string_view program) {
    return std::string(program) + " " + ninfer::build_info::kVersion + " (" +
           ninfer::build_info::kGitSha + "; " + ninfer::build_info::kBuildType + "; " +
           ninfer::build_info::kCompiler + "; " + ninfer::build_info::kCudaArch + ")";
}

} // namespace ninfer::product
