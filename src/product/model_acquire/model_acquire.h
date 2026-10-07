#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::product::model_acquire {

enum class ErrorKind {
    InvalidSpec,
    OfflineNotCached,
    DownloadFailed,
    NotFound,
    CacheError,
};

class Error final : public std::runtime_error {
public:
    Error(ErrorKind kind, std::string message)
        : std::runtime_error(std::move(message)), kind_(kind) {}

    [[nodiscard]] ErrorKind kind() const noexcept { return kind_; }

private:
    ErrorKind kind_;
};

struct HfSpec {
    std::string repo;
    std::string file;
    std::string revision = "main";
    // Explicit `--hf-token` value; empty means "fall back to the environment".
    std::string token;
};

struct Acquisition {
    std::optional<HfSpec> hf;
    std::optional<std::string> model_url;
    // Empty means `default_cache_dir()`.
    std::filesystem::path cache_dir;
    bool offline    = false;
    bool cache_list = false;
};

struct CacheEntry {
    std::filesystem::path relative_path;
    std::uintmax_t size_bytes = 0;
};

[[nodiscard]] std::filesystem::path default_cache_dir();
[[nodiscard]] std::string resolve_hf_token(const std::string& explicit_token);
[[nodiscard]] std::string hf_download_url(const HfSpec& spec);
[[nodiscard]] std::filesystem::path cache_path_for_hf(const std::filesystem::path& cache_dir,
                                                      const HfSpec& spec);
[[nodiscard]] std::filesystem::path cache_path_for_url(const std::filesystem::path& cache_dir,
                                                       std::string_view url);

void validate_hf_spec(const HfSpec& spec);
void validate_model_url(std::string_view url);

[[nodiscard]] std::vector<CacheEntry> list_cache(const std::filesystem::path& cache_dir);

struct ResolveResult {
    std::filesystem::path artifact_path;
    bool from_cache = false;
    bool downloaded = false;
};

// Slice 0: resolves local paths, cache hits, `--offline` and `--cache-list`.
// A remote source whose file is not yet cached fails closed with DownloadFailed;
// the streaming download lands in slice 1.
[[nodiscard]] ResolveResult resolve_local(const Acquisition& acquisition,
                                         const std::filesystem::path& positional_artifact);

} // namespace ninfer::product::model_acquire
