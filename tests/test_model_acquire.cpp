#include "product/model_acquire/model_acquire.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>

#include <unistd.h>

namespace {

namespace acquire = ninfer::product::model_acquire;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool rejects_with(const std::function<void()>& operation, std::string_view needle) {
    try {
        operation();
    } catch (const acquire::Error& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    } catch (const std::invalid_argument& error) {
        return std::string(error.what()).find(needle) != std::string::npos;
    }
    return false;
}

struct EnvGuard {
    std::string name;
    std::string saved;
    bool had = false;
    explicit EnvGuard(std::string n) : name(std::move(n)) {
        if (const char* value = std::getenv(name.c_str())) {
            saved = value;
            had   = true;
        }
    }
    ~EnvGuard() {
        if (had) {
            ::setenv(name.c_str(), saved.c_str(), 1);
        } else {
            ::unsetenv(name.c_str());
        }
    }
};

std::filesystem::path unique_temp(const char* tag) {
    const auto base = std::filesystem::temp_directory_path() /
                      ("ninfer_acquire_" + std::string(tag) + "_" + std::to_string(::getpid()));
    std::error_code ignored;
    std::filesystem::remove_all(base, ignored);
    std::filesystem::create_directories(base, ignored);
    return base;
}

void write_file(const std::filesystem::path& path, std::string_view content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << content;
}

} // namespace

int main() {
    int failures = 0;

    {
        EnvGuard frinfer("FRINFER_CACHE_DIR");
        EnvGuard xdg("XDG_CACHE_HOME");
        EnvGuard home("HOME");
        ::setenv("FRINFER_CACHE_DIR", "/tmp/custom-cache", 1);
        ::unsetenv("XDG_CACHE_HOME");
        failures += check(acquire::default_cache_dir() == "/tmp/custom-cache",
                          "FRINFER_CACHE_DIR override was not honored");
        ::unsetenv("FRINFER_CACHE_DIR");
        ::setenv("XDG_CACHE_HOME", "/tmp/xdg", 1);
        failures += check(acquire::default_cache_dir() == "/tmp/xdg/frinfer",
                          "XDG_CACHE_HOME default was not honored");
    }

    failures += check(acquire::resolve_hf_token("explicit") == "explicit",
                      "explicit token was not preserved");
    {
        EnvGuard hf("HF_TOKEN");
        EnvGuard hub("HUGGING_FACE_HUB_TOKEN");
        ::unsetenv("HF_TOKEN");
        ::unsetenv("HUGGING_FACE_HUB_TOKEN");
        failures += check(acquire::resolve_hf_token("").empty(),
                          "empty token did not stay empty without env");
        ::setenv("HF_TOKEN", "env-token", 1);
        failures += check(acquire::resolve_hf_token("") == "env-token",
                          "HF_TOKEN env fallback failed");
    }

    const acquire::HfSpec spec{.repo     = "neroued/Qwen3.8-27B-nvfp4-NInfer",
                               .file     = "qwen3_8_27b_nvfp4.ninfer",
                               .revision = "main"};
    failures +=
        check(acquire::hf_download_url(spec) ==
                  "https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/resolve/main/"
                  "qwen3_8_27b_nvfp4.ninfer",
              "HF download URL mismatch");
    failures += check(acquire::cache_path_for_hf("/cache", spec) ==
                          "/cache/hf/neroued-Qwen3.8-27B-nvfp4-NInfer/main/"
                          "qwen3_8_27b_nvfp4.ninfer",
                      "HF cache path mismatch");
    failures += check(acquire::cache_path_for_url(
                                        "/cache", "https://example.com/models/model.ninfer") ==
                                          "/cache/urls/" +
                                              acquire::cache_path_for_url(
                                                  "/cache",
                                                  "https://example.com/models/model.ninfer")
                                                  .parent_path()
                                                  .filename()
                                                  .string() +
                                              "/model.ninfer",
                      "URL cache path is not deterministic");

    failures += check(
        rejects_with([] {
            acquire::validate_hf_spec(
                {.repo = "no-slash", .file = "model.ninfer", .revision = "main"});
        }, "hf_spec_invalid"),
        "bare HF repo was accepted");
    failures += check(
        rejects_with([] {
            acquire::validate_hf_spec(
                {.repo = "a/b/c", .file = "model.ninfer", .revision = "main"});
        }, "hf_spec_invalid"),
        "double-slash HF repo was accepted");
    failures += check(
        rejects_with([] {
            acquire::validate_hf_spec(
                {.repo = "owner/repo", .file = "dir/model.ninfer", .revision = "main"});
        }, "--hf-file"),
        "HF file path was accepted");
    failures += check(
        rejects_with([] {
            acquire::validate_hf_spec(
                {.repo = "owner/repo", .file = "model.bin", .revision = "main"});
        }, ".ninfer"),
        "non-.ninfer HF file was accepted");
    failures += check(rejects_with([] { acquire::validate_model_url("http://x/model.ninfer"); },
                                   "model_url_invalid"),
                      "http model URL was accepted");
    failures += check(
        rejects_with([] { acquire::validate_model_url("https://x/model.bin"); }, ".ninfer"),
        "non-.ninfer model URL was accepted");

    {
        const auto tmp = unique_temp("list");
        failures += check(acquire::list_cache(tmp / "missing").empty(),
                          "missing cache dir did not list empty");
        write_file(tmp / "hf" / "owner-repo" / "main" / "a.ninfer", "weights");
        write_file(tmp / "hf" / "owner-repo" / "main" / "a.ninfer.part", "partial");
        write_file(tmp / "notes.txt", "ignore");
        const auto entries = acquire::list_cache(tmp);
        failures += check(entries.size() == 1 &&
                              entries[0].relative_path ==
                                  "hf/owner-repo/main/a.ninfer" &&
                              entries[0].size_bytes == 7,
                          "cache listing did not return the single .ninfer entry");
        std::error_code ignored;
        std::filesystem::remove_all(tmp, ignored);
    }

    {
        const auto tmp = unique_temp("resolve");
        acquire::Acquisition local;
        const auto passthrough =
            acquire::resolve_local(local, tmp / "model.ninfer");
        failures += check(passthrough.artifact_path == tmp / "model.ninfer" &&
                              !passthrough.from_cache && !passthrough.downloaded,
                          "local path did not pass through");

        acquire::Acquisition both;
        both.hf        = spec;
        both.model_url = "https://example.com/models/model.ninfer";
        failures += check(rejects_with([&] { (void)acquire::resolve_local(both, {}); },
                                       "mutually exclusive"),
                          "hf+url combination was accepted");

        acquire::Acquisition neither;
        failures += check(rejects_with([&] { (void)acquire::resolve_local(neither, {}); },
                                       ".ninfer model path is required"),
                          "empty source was accepted");

        const auto cached_path = acquire::cache_path_for_hf(tmp, spec);
        write_file(cached_path, "cached-weights");
        acquire::Acquisition hit;
        hit.hf        = spec;
        hit.cache_dir = tmp;
        const auto resolved = acquire::resolve_local(hit, {});
        failures += check(resolved.artifact_path == cached_path && resolved.from_cache &&
                              !resolved.downloaded,
                          "HF cache hit did not resolve");

        acquire::Acquisition miss;
        miss.hf        = spec;
        miss.cache_dir = tmp / "empty";
        miss.offline   = true;
        failures += check(rejects_with([&] { (void)acquire::resolve_local(miss, {}); },
                                       "offline_not_cached"),
                          "offline cache miss omitted offline_not_cached");

        acquire::Acquisition pending;
        pending.hf        = spec;
        pending.cache_dir = tmp / "empty";
        failures += check(rejects_with([&] { (void)acquire::resolve_local(pending, {}); },
                                       "hf_download_failed"),
                          "uncached remote did not fail closed");

        std::error_code ignored;
        std::filesystem::remove_all(tmp, ignored);
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
