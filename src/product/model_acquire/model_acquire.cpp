#include "product/model_acquire/model_acquire.h"

#include <curl/curl.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>

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


namespace download {
namespace {

using Clock = std::chrono::steady_clock;

void ensure_curl_init() {
    static std::once_flag init;
    std::call_once(init, [] {
        if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
            throw Error(ErrorKind::DownloadFailed,
                        "hf_download_failed: failed to initialize libcurl");
        }
    });
}

bool private_ipv4(std::uint32_t address) {
    const std::uint32_t a = ntohl(address);
    return (a >> 24U) == 0 || (a >> 24U) == 10 || (a >> 24U) == 127 || (a >> 16U) == 0xa9fe ||
           (a >> 20U) == 0xac1 || (a >> 16U) == 0xc0a8 || (a >> 22U) == 0x0191 ||
           (a >> 17U) == 0x6309 || (a >> 24U) >= 224;
}

bool private_address(const sockaddr* address) {
    if (address->sa_family == AF_INET) {
        return private_ipv4(reinterpret_cast<const sockaddr_in*>(address)->sin_addr.s_addr);
    }
    if (address->sa_family != AF_INET6) { return true; }
    const in6_addr& a = reinterpret_cast<const sockaddr_in6*>(address)->sin6_addr;
    if (IN6_IS_ADDR_UNSPECIFIED(&a) || IN6_IS_ADDR_LOOPBACK(&a) || IN6_IS_ADDR_LINKLOCAL(&a) ||
        IN6_IS_ADDR_MULTICAST(&a) || (a.s6_addr[0] & 0xfeU) == 0xfcU) {
        return true;
    }
    if (IN6_IS_ADDR_V4MAPPED(&a)) {
        std::uint32_t v4 = 0;
        std::memcpy(&v4, &a.s6_addr[12], sizeof(v4));
        return private_ipv4(v4);
    }
    return false;
}

struct UrlParts {
    std::string scheme;
    std::string host;
    std::string port;
};

std::string curlu_part(CURLU* url, CURLUPart part, unsigned flags = 0) {
    char* raw            = nullptr;
    const CURLUcode code = curl_url_get(url, part, &raw, flags);
    if (code != CURLUE_OK || raw == nullptr) { return {}; }
    std::string out(raw);
    curl_free(raw);
    return out;
}

UrlParts parse_https_url(std::string_view value, std::string_view what) {
    std::unique_ptr<CURLU, decltype(&curl_url_cleanup)> url(curl_url(), curl_url_cleanup);
    if (!url || curl_url_set(url.get(), CURLUPART_URL, std::string(value).c_str(), 0) != CURLUE_OK) {
        throw Error(ErrorKind::InvalidSpec,
                    std::string(what) + " is not a valid URL: " + std::string(value));
    }
    UrlParts out;
    out.scheme = curlu_part(url.get(), CURLUPART_SCHEME);
    out.host   = curlu_part(url.get(), CURLUPART_HOST);
    out.port   = curlu_part(url.get(), CURLUPART_PORT, CURLU_DEFAULT_PORT);
    const std::string user = curlu_part(url.get(), CURLUPART_USER);
    if (out.scheme != "https" || out.host.empty() || out.port.empty() || !user.empty()) {
        throw Error(ErrorKind::DownloadFailed,
                    "hf_download_failed: redirect left https or carried credentials");
    }
    return out;
}

std::string resolve_public_ip(const UrlParts& parts) {
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* raw     = nullptr;
    const int rc      = getaddrinfo(parts.host.c_str(), parts.port.c_str(), &hints, &raw);
    if (rc != 0) {
        throw Error(ErrorKind::DownloadFailed,
                    "hf_download_failed: DNS resolution failed for " + parts.host + ": " +
                        gai_strerror(rc));
    }
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses(raw, freeaddrinfo);
    for (const addrinfo* it = raw; it != nullptr; it = it->ai_next) {
        if (private_address(it->ai_addr)) { continue; }
        std::array<char, INET6_ADDRSTRLEN> text{};
        const void* bytes =
            it->ai_family == AF_INET
                ? static_cast<const void*>(
                      &reinterpret_cast<const sockaddr_in*>(it->ai_addr)->sin_addr)
                : static_cast<const void*>(
                      &reinterpret_cast<const sockaddr_in6*>(it->ai_addr)->sin6_addr);
        if (inet_ntop(it->ai_family, bytes, text.data(), text.size()) != nullptr) {
            return text.data();
        }
    }
    throw Error(ErrorKind::DownloadFailed,
                "hf_download_failed: " + parts.host + " resolves only to private addresses");
}

struct Throttle {
    DownloadProgress callback;
    Clock::time_point last = Clock::now();
};

int xfer_info(void* opaque, curl_off_t total, curl_off_t now, curl_off_t, curl_off_t) noexcept {
    auto* state = static_cast<Throttle*>(opaque);
    if (!state->callback) { return 0; }
    const auto elapsed = Clock::now() - state->last;
    if (elapsed < std::chrono::milliseconds(500) && now != total) { return 0; }
    state->last = Clock::now();
    try {
        state->callback(static_cast<std::uint64_t>(now), static_cast<std::uint64_t>(total));
    } catch (...) {
        return 1;
    }
    return 0;
}

std::size_t write_file(char* data, std::size_t size, std::size_t count, void* opaque) {
    auto* stream = static_cast<std::ofstream*>(opaque);
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size) { return 0; }
    stream->write(data, static_cast<std::streamsize>(size * count));
    return stream->good() ? size * count : 0;
}

struct Performed {
    CURLcode code;
    long status         = 0;
    std::string redirect;
};

Performed perform_one(const std::string& url, const UrlParts& /*parts*/, const std::string& resolve,
                      std::ofstream& stream, const std::string& token,
                      Throttle& throttle, curl_off_t resume_from) {
    std::unique_ptr<CURL, decltype(&curl_easy_cleanup)> curl(curl_easy_init(), curl_easy_cleanup);
    if (!curl) { throw Error(ErrorKind::DownloadFailed, "hf_download_failed: no curl handle"); }
    curl_slist* headers = nullptr;
    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> header_guard(
        nullptr, curl_slist_free_all);
    std::string auth;
    std::string range;
    if (!token.empty()) {
        auth    = "Authorization: Bearer " + token;
        headers = curl_slist_append(headers, auth.c_str());
    }
    if (resume_from > 0) {
        // A manual Range header, mirroring `curl -r`: unlike CURLOPT_RESUME_FROM_LARGE
        // it never turns a redirect hop into a client-side range error.
        range   = "Range: bytes=" + std::to_string(resume_from) + "-";
        headers = curl_slist_append(headers, range.c_str());
    }
    header_guard.reset(headers);
    curl_slist* resolve_list = curl_slist_append(nullptr, resolve.c_str());
    if (resolve_list == nullptr) { throw std::bad_alloc(); }
    std::unique_ptr<curl_slist, decltype(&curl_slist_free_all)> resolve_guard(resolve_list,
                                                                               curl_slist_free_all);
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_PROTOCOLS_STR, "https");
    curl_easy_setopt(curl.get(), CURLOPT_REDIR_PROTOCOLS_STR, "https");
    curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 0L);
    curl_easy_setopt(curl.get(), CURLOPT_CONNECTTIMEOUT_MS, 15000L);
    curl_easy_setopt(curl.get(), CURLOPT_LOW_SPEED_LIMIT, 512L * 1024L);
    curl_easy_setopt(curl.get(), CURLOPT_LOW_SPEED_TIME, 120L);
    curl_easy_setopt(curl.get(), CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl.get(), CURLOPT_SSL_VERIFYHOST, 2L);
    curl_easy_setopt(curl.get(), CURLOPT_PROXY, "");
    curl_easy_setopt(curl.get(), CURLOPT_RESOLVE, resolve_list);
    if (headers != nullptr) { curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers); }
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, write_file);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &stream);
    curl_easy_setopt(curl.get(), CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl.get(), CURLOPT_XFERINFOFUNCTION, xfer_info);
    curl_easy_setopt(curl.get(), CURLOPT_XFERINFODATA, &throttle);
    curl_easy_setopt(curl.get(), CURLOPT_USERAGENT, "frinfer/model-acquire");
    const CURLcode code = curl_easy_perform(curl.get());
    long status         = 0;
    std::string redirect;
    if (code == CURLE_OK) {
        curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &status);
        char* next = nullptr;
        curl_easy_getinfo(curl.get(), CURLINFO_REDIRECT_URL, &next);
        if (next != nullptr && *next != '\0') { redirect = next; }
    }
    return {.code = code, .status = status, .redirect = std::move(redirect)};
}

void fail_status(long status, const std::string& source, const std::string& token) {
    if (status == 404 || status == 410) {
        throw Error(ErrorKind::NotFound,
                    "hf_not_found: " + source + " returned HTTP " + std::to_string(status) +
                        " (check --hf-repo/--hf-file/--hf-revision or the file name)");
    }
    if ((status == 401 || status == 403) && !token.empty()) {
        throw Error(ErrorKind::DownloadFailed,
                    "hf_download_failed: HTTP " + std::to_string(status) +
                        " (the provided --hf-token was rejected)");
    }
    if (status == 401 || status == 403) {
        throw Error(ErrorKind::DownloadFailed,
                    "hf_download_failed: HTTP " + std::to_string(status) +
                        " (this file may need --hf-token)");
    }
    throw Error(ErrorKind::DownloadFailed,
                "hf_download_failed: " + source + " returned HTTP " + std::to_string(status));
}

void download_to(const std::string& initial_url, const std::filesystem::path& dest,
                 const std::string& token, const DownloadProgress& progress,
                 std::string_view source_label) {
    ensure_curl_init();
    std::error_code error;
    std::filesystem::create_directories(dest.parent_path(), error);
    if (error) {
        throw Error(ErrorKind::CacheError,
                    "hf_download_failed: cannot create cache dir: " + error.message());
    }
    const std::filesystem::path tmp = dest.string() + ".part";
    std::string url(initial_url);
    Throttle throttle{.callback = progress};
    bool retried_full = false;
    for (int hop = 0; hop <= 5; ++hop) {
        const UrlParts parts = parse_https_url(url, source_label);
        const std::string ip =
            resolve_public_ip(parts); // fail closed on private-only resolution
        std::string resolve = parts.host + ":" + parts.port + ":";
        resolve += ip.find(':') == std::string::npos ? ip : "[" + ip + "]";

        std::uintmax_t existing = 0;
        {
            std::error_code size_error;
            if (std::filesystem::is_regular_file(tmp, size_error)) {
                existing = std::filesystem::file_size(tmp, size_error);
            }
        }
        std::ofstream stream(tmp, std::ios::binary | std::ios::app);
        if (!stream) {
            throw Error(ErrorKind::CacheError,
                        "hf_download_failed: cannot write " + tmp.string());
        }
        const Performed done = perform_one(url, parts, resolve, stream,
                                           token, throttle,
                                           static_cast<curl_off_t>(existing));
        stream.flush();
        stream.close();
        if (done.code != CURLE_OK) {
            if (done.code == CURLE_OPERATION_TIMEDOUT) {
                throw Error(ErrorKind::DownloadFailed,
                            "hf_download_failed: transfer timed out; rerun to resume " +
                                tmp.string());
            }
            throw Error(ErrorKind::DownloadFailed,
                        std::string("hf_download_failed: ") + curl_easy_strerror(done.code) +
                            "; rerun to resume " + tmp.string());
        }
        if (done.status == 200 || done.status == 206) {
            if (done.status == 200 && existing > 0 && !retried_full) {
                // Server ignored the Range resume: restart once from zero.
                retried_full = true;
                std::filesystem::resize_file(tmp, 0, error);
                url = initial_url;
                hop = -1;
                continue;
            }
            std::filesystem::rename(tmp, dest, error);
            if (error) {
                throw Error(ErrorKind::CacheError,
                            "hf_download_failed: cannot publish " + dest.string() + ": " +
                                error.message());
            }
            if (progress) {
                std::error_code size_error;
                const auto total =
                    std::filesystem::file_size(dest, size_error);
                progress(size_error ? 0 : static_cast<std::uint64_t>(total),
                         size_error ? 0 : static_cast<std::uint64_t>(total));
            }
            return;
        }
        if (done.status == 301 || done.status == 302 || done.status == 303 ||
            done.status == 307 || done.status == 308) {
            if (done.redirect.empty()) {
                throw Error(ErrorKind::DownloadFailed,
                            "hf_download_failed: redirect without Location for " + url);
            }
            url = std::move(done.redirect);
            continue;
        }
        if (done.status == 416 && !retried_full) {
            retried_full = true;
            std::filesystem::resize_file(tmp, 0, error);
            url = initial_url;
            hop = -1;
            continue;
        }
        fail_status(done.status, std::string(source_label), token);
    }
    throw Error(ErrorKind::DownloadFailed, "hf_download_failed: too many redirects for " + url);
}

} // namespace
} // namespace download

std::filesystem::path download_hf(const HfSpec& spec, const std::filesystem::path& cache_dir,
                                   const DownloadProgress& progress) {
    validate_hf_spec(spec);
    const std::filesystem::path base =
        cache_dir.empty() ? default_cache_dir() : cache_dir;
    const std::filesystem::path dest = cache_path_for_hf(base, spec);
    std::error_code error;
    if (std::filesystem::is_regular_file(dest, error)) { return dest; }
    download::download_to(hf_download_url(spec), dest, resolve_hf_token(spec.token), progress,
                spec.repo + "/" + spec.file);
    return dest;
}

std::filesystem::path download_url(std::string_view url, const std::filesystem::path& cache_dir,
                                    const DownloadProgress& progress) {
    validate_model_url(url);
    const std::filesystem::path base =
        cache_dir.empty() ? default_cache_dir() : cache_dir;
    const std::filesystem::path dest = cache_path_for_url(base, url);
    std::error_code error;
    if (std::filesystem::is_regular_file(dest, error)) { return dest; }
    download::download_to(std::string(url), dest, {}, progress, url);
    return dest;
}

ResolveResult resolve(const Acquisition& acquisition,
                      const std::filesystem::path& positional_artifact,
                      const DownloadProgress& progress) {
    if (acquisition.hf && acquisition.model_url) {
        throw Error(ErrorKind::InvalidSpec,
                    "hf_spec_invalid: --hf-repo and --model-url are mutually exclusive");
    }
    const std::filesystem::path cache_dir =
        acquisition.cache_dir.empty() ? default_cache_dir() : acquisition.cache_dir;
    if (acquisition.hf) {
        validate_hf_spec(*acquisition.hf);
        const std::filesystem::path cached = cache_path_for_hf(cache_dir, *acquisition.hf);
        std::error_code error;
        if (std::filesystem::is_regular_file(cached, error)) {
            return {.artifact_path = cached, .from_cache = true, .downloaded = false};
        }
        if (acquisition.offline) {
            throw Error(ErrorKind::OfflineNotCached,
                        "offline_not_cached: " + acquisition.hf->repo + "/" +
                            acquisition.hf->file + " is not in " + cache_dir.string() +
                            "; download it first without --offline");
        }
        return {.artifact_path = download_hf(*acquisition.hf, cache_dir, progress),
                .from_cache    = false,
                .downloaded    = true};
    }
    if (acquisition.model_url) {
        validate_model_url(*acquisition.model_url);
        const std::filesystem::path cached = cache_path_for_url(cache_dir, *acquisition.model_url);
        std::error_code error;
        if (std::filesystem::is_regular_file(cached, error)) {
            return {.artifact_path = cached, .from_cache = true, .downloaded = false};
        }
        if (acquisition.offline) {
            throw Error(ErrorKind::OfflineNotCached,
                        "offline_not_cached: " + *acquisition.model_url + " is not in " +
                            cache_dir.string() + "; download it first without --offline");
        }
        return {.artifact_path = download_url(*acquisition.model_url, cache_dir, progress),
                .from_cache    = false,
                .downloaded    = true};
    }
    if (!positional_artifact.empty()) {
        return {.artifact_path = positional_artifact, .from_cache = false, .downloaded = false};
    }
    if (acquisition.cache_list) {
        throw Error(ErrorKind::InvalidSpec, ".ninfer model path is required");
    }
    throw Error(ErrorKind::InvalidSpec, ".ninfer model path is required");
}

} // namespace ninfer::product::model_acquire
