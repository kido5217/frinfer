#include "product/model_acquire/model_acquire.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <sstream>

namespace ninfer::product::model_acquire {
namespace {

bool has_control(std::string_view text) {
    return std::any_of(text.begin(), text.end(), [](char c) {
        return std::iscntrl(static_cast<unsigned char>(c)) != 0;
    });
}

void require_no_dotdot(std::string_view text, std::string_view what) {
    if (text.find("..") != std::string_view::npos) {
        throw Error(ErrorKind::InvalidSpec,
                    std::string(what) + " must not contain '..': " + std::string(text));
    }
    if (text.find('\\') != std::string_view::npos) {
        throw Error(ErrorKind::InvalidSpec,
                    std::string(what) + " must not contain backslashes: " + std::string(text));
    }
    if (has_control(text)) {
        throw Error(ErrorKind::InvalidSpec,
                    std::string(what) + " must not contain control characters");
    }
}

void require_hf_token(std::string_view text) {
    if (text.empty() || text.size() > 512 || has_control(text)) {
        throw Error(ErrorKind::InvalidSpec, "hf_spec_invalid: --hf-repo must be OWNER/REPO");
    }
    const std::size_t slash = text.find('/');
    if (slash == std::string_view::npos || slash == 0 || slash + 1 >= text.size() ||
        text.find('/', slash + 1) != std::string_view::npos) {
        throw Error(ErrorKind::InvalidSpec, "hf_spec_invalid: --hf-repo must be OWNER/REPO");
    }
    require_no_dotdot(text, "--hf-repo");
}

void require_file_name(std::string_view text, std::string_view flag) {
    if (text.empty() || text.size() > 512) {
        throw Error(ErrorKind::InvalidSpec,
                    std::string(flag) + " must be a non-empty file name");
    }
    if (text.front() == '/' || text.find('/') != std::string_view::npos) {
        throw Error(ErrorKind::InvalidSpec,
                    std::string(flag) + " must be a bare file name, not a path");
    }
    require_no_dotdot(text, flag);
    if (!text.ends_with(".ninfer")) {
        throw Error(ErrorKind::InvalidSpec,
                    std::string(flag) + " must name a .ninfer artifact: " + std::string(text));
    }
}

void require_revision(std::string_view text) {
    if (text.empty() || text.size() > 128) {
        throw Error(ErrorKind::InvalidSpec, "hf_spec_invalid: --hf-revision must not be empty");
    }
    if (text.find('/') != std::string_view::npos) {
        throw Error(ErrorKind::InvalidSpec,
                    "hf_spec_invalid: --hf-revision must not contain '/': " + std::string(text));
    }
    require_no_dotdot(text, "--hf-revision");
}

std::string sanitized_repo(std::string_view repo) {
    std::string out(repo);
    std::replace(out.begin(), out.end(), '/', '-');
    std::replace(out.begin(), out.end(), ' ', '_');
    return out;
}

std::string fnv1a_hex(std::string_view text) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const unsigned char c : text) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}

std::string url_basename(std::string_view url) {
    const std::size_t query = url.find_first_of("?#");
    const std::string_view clean =
        query == std::string_view::npos ? url : url.substr(0, query);
    const std::size_t slash = clean.rfind('/');
    const std::string_view base =
        slash == std::string_view::npos ? clean : clean.substr(slash + 1);
    return std::string(base);
}

bool cached_file_present(const std::filesystem::path& path) {
    std::error_code error;
    return std::filesystem::is_regular_file(path, error);
}

} // namespace

std::filesystem::path default_cache_dir() {
    if (const char* override = std::getenv("FRINFER_CACHE_DIR");
        override != nullptr && *override != '\0') {
        return std::filesystem::path(override);
    }
    if (const char* xdg = std::getenv("XDG_CACHE_HOME");
        xdg != nullptr && *xdg != '\0') {
        return std::filesystem::path(xdg) / "frinfer";
    }
    if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
        return std::filesystem::path(home) / ".cache" / "frinfer";
    }
    return std::filesystem::temp_directory_path() / "frinfer-cache";
}

std::string resolve_hf_token(const std::string& explicit_token) {
    if (!explicit_token.empty()) { return explicit_token; }
    if (const char* token = std::getenv("HF_TOKEN");
        token != nullptr && *token != '\0') {
        return token;
    }
    if (const char* token = std::getenv("HUGGING_FACE_HUB_TOKEN");
        token != nullptr && *token != '\0') {
        return token;
    }
    return {};
}

void validate_hf_spec(const HfSpec& spec) {
    require_hf_token(spec.repo);
    require_file_name(spec.file, "--hf-file");
    require_revision(spec.revision);
    if (spec.token.size() > 4096 || has_control(spec.token)) {
        throw Error(ErrorKind::InvalidSpec, "hf_spec_invalid: --hf-token is malformed");
    }
}

void validate_model_url(std::string_view url) {
    if (url.size() > 2048 || url.substr(0, 8) != "https://") {
        throw Error(ErrorKind::InvalidSpec,
                    "model_url_invalid: --model-url must be an https:// URL");
    }
    if (has_control(url)) {
        throw Error(ErrorKind::InvalidSpec, "model_url_invalid: URL has control characters");
    }
    require_no_dotdot(url_basename(url), "--model-url file");
    require_file_name(url_basename(url), "--model-url");
}

std::string hf_download_url(const HfSpec& spec) {
    validate_hf_spec(spec);
    return "https://huggingface.co/" + spec.repo + "/resolve/" + spec.revision + "/" + spec.file;
}

std::filesystem::path cache_path_for_hf(const std::filesystem::path& cache_dir,
                                        const HfSpec& spec) {
    validate_hf_spec(spec);
    const std::filesystem::path base =
        cache_dir.empty() ? default_cache_dir() : cache_dir;
    return base / "hf" / sanitized_repo(spec.repo) / spec.revision / spec.file;
}

std::filesystem::path cache_path_for_url(const std::filesystem::path& cache_dir,
                                         std::string_view url) {
    validate_model_url(url);
    const std::filesystem::path base =
        cache_dir.empty() ? default_cache_dir() : cache_dir;
    return base / "urls" / fnv1a_hex(url) / url_basename(url);
}

std::vector<CacheEntry> list_cache(const std::filesystem::path& cache_dir) {
    const std::filesystem::path base =
        cache_dir.empty() ? default_cache_dir() : cache_dir;
    std::vector<CacheEntry> entries;
    std::error_code error;
    if (!std::filesystem::is_directory(base, error)) { return entries; }
    for (std::filesystem::recursive_directory_iterator it(base, error), end;
         !error && it != end; it.increment(error)) {
        if (!it->is_regular_file(error)) { continue; }
        const std::filesystem::path full = it->path();
        const std::string name           = full.filename().string();
        if (name.ends_with(".part") || name.ends_with(".tmp")) { continue; }
        if (full.extension() != ".ninfer") { continue; }
        CacheEntry entry;
        entry.relative_path = std::filesystem::relative(full, base, error);
        if (error) { continue; }
        entry.size_bytes = it->file_size(error);
        if (error) { continue; }
        entries.push_back(std::move(entry));
    }
    std::sort(entries.begin(), entries.end(), [](const CacheEntry& a, const CacheEntry& b) {
        return a.relative_path < b.relative_path;
    });
    return entries;
}

ResolveResult resolve_local(const Acquisition& acquisition,
                            const std::filesystem::path& positional_artifact) {
    const std::filesystem::path cache_dir =
        acquisition.cache_dir.empty() ? default_cache_dir() : acquisition.cache_dir;

    if (acquisition.hf && acquisition.model_url) {
        throw Error(ErrorKind::InvalidSpec,
                    "hf_spec_invalid: --hf-repo and --model-url are mutually exclusive");
    }
    if (acquisition.hf) { validate_hf_spec(*acquisition.hf); }
    if (acquisition.model_url) { validate_model_url(*acquisition.model_url); }

    const bool has_remote = acquisition.hf.has_value() || acquisition.model_url.has_value();
    if (has_remote && !positional_artifact.empty()) {
        throw Error(ErrorKind::InvalidSpec,
                    "hf_spec_invalid: positional <model.ninfer> and --hf-repo/--model-url "
                    "are mutually exclusive");
    }
    if (!has_remote && !acquisition.cache_list && positional_artifact.empty()) {
        throw Error(ErrorKind::InvalidSpec, ".ninfer model path is required");
    }

    if (acquisition.hf) {
        const std::filesystem::path cached = cache_path_for_hf(cache_dir, *acquisition.hf);
        if (cached_file_present(cached)) {
            return {.artifact_path = cached, .from_cache = true, .downloaded = false};
        }
        if (acquisition.offline) {
            throw Error(ErrorKind::OfflineNotCached,
                        "offline_not_cached: " + acquisition.hf->repo + "/" +
                            acquisition.hf->file + " is not in " + cache_dir.string() +
                            "; download it first without --offline");
        }
        throw Error(ErrorKind::DownloadFailed,
                    "hf_download_failed: download not implemented in this build slice; "
                    "wait for the streaming download slice");
    }
    if (acquisition.model_url) {
        const std::filesystem::path cached = cache_path_for_url(cache_dir, *acquisition.model_url);
        if (cached_file_present(cached)) {
            return {.artifact_path = cached, .from_cache = true, .downloaded = false};
        }
        if (acquisition.offline) {
            throw Error(ErrorKind::OfflineNotCached,
                        "offline_not_cached: " + *acquisition.model_url + " is not in " +
                            cache_dir.string() + "; download it first without --offline");
        }
        throw Error(ErrorKind::DownloadFailed,
                    "hf_download_failed: download not implemented in this build slice; "
                    "wait for the streaming download slice");
    }
    return {.artifact_path = positional_artifact, .from_cache = false, .downloaded = false};
}

} // namespace ninfer::product::model_acquire
