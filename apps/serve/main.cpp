#include "product/build_info/build_info.h"
#include "product/model_acquire/model_acquire.h"
#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

} // namespace

int main(int argc, char** argv) {
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "frinfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "frinfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }
    if (options.version_requested) {
        std::cout << ninfer::product::version_text("frinfer-serve") << '\n';
        return 0;
    }

    if (options.acquisition.cache_list) {
        try {
            const std::filesystem::path base =
                options.acquisition.cache_dir.empty()
                    ? ninfer::product::model_acquire::default_cache_dir()
                    : options.acquisition.cache_dir;
            for (const auto& entry :
                 ninfer::product::model_acquire::list_cache(base)) {
                std::cout << entry.relative_path.string() << ' ' << entry.size_bytes << '\n';
            }
            return 0;
        } catch (const std::exception& error) {
            std::cerr << "frinfer-serve: " << error.what() << '\n';
            return 1;
        }
    }
    try {
        ninfer::product::model_acquire::Acquisition request;
        if (options.acquisition.hf_repo) {
            ninfer::product::model_acquire::HfSpec hf;
            hf.repo     = *options.acquisition.hf_repo;
            hf.file     = options.acquisition.hf_file.value_or("");
            hf.revision = options.acquisition.hf_revision.value_or("main");
            hf.token    = options.acquisition.hf_token.value_or("");
            request.hf  = std::move(hf);
        }
        if (options.acquisition.model_url) { request.model_url = options.acquisition.model_url; }
        request.cache_dir = options.acquisition.cache_dir;
        request.offline   = options.acquisition.offline;
        const std::filesystem::path positional =
            options.artifact_path.empty() ? std::filesystem::path{}
                                          : std::filesystem::path(options.artifact_path);
        auto progress = [](std::uint64_t done, std::uint64_t total) {
            std::cerr << "download " << done / (1ULL << 20) << " MiB";
            if (total > 0) {
                std::cerr << " / " << total / (1ULL << 20) << " MiB ("
                          << (100ULL * done / total) << "%)";
            }
            std::cerr << '\n';
        };
        const auto resolved = ninfer::product::model_acquire::resolve(
            request, positional, progress);
        options.artifact_path = resolved.artifact_path.string();
        if (resolved.downloaded) {
            std::cerr << "downloaded " << options.artifact_path << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "frinfer-serve: " << error.what() << '\n';
        return 1;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "frinfer-serve",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    bool serving = false;

    try {
        ninfer::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        ninfer::serve::GenerationService service(options, startup_log.observer());
        startup_log.engine_ready(service.load_summary());
        operational_log.engine_capacity(service);

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        g_server.store(nullptr);
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
