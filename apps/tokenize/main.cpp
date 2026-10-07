// frinfer-tokenize: the artifact's tokenizer, exposed. Encode text to token ids, decode ids to
// text, and report each id's spelling and special-token status.
//
// This is the debugging primitive for prompt framing, chat-template drift, and token budgets. It
// runs on the same public Engine load route as the CLI and perplexity app; there is no
// tokenizer-only artifact load path.

#include "product/build_info/build_info.h"
#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/logging/startup_log.h"

#include "ninfer/engine.h"

#include <spdlog/logger.h>

#include <charconv>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using ninfer::product::LogLevel;

enum class Mode : std::uint8_t {
    // Text in, ids out.
    Encode,
    // Ids in, text out.
    Decode,
};

struct Options {
    bool help_requested    = false;
    bool version_requested = false;
    std::filesystem::path artifact;
    Mode mode              = Mode::Encode;

    // Encode input. Exactly one of --text, --file or --stdin.
    std::string text;
    std::filesystem::path file;
    bool stdin_source = false;

    // Decode input: whitespace/comma-separated ids, inline or from a file.
    std::string ids;
    std::filesystem::path ids_file;

    // Output shape.
    bool show_pieces        = false;
    bool show_count         = false;
    bool ids_only           = false;
    bool decode_text        = false;
    bool skip_special_tokens = false;

    int device      = 0;
    LogLevel log_level = LogLevel::Info;
};

std::string usage_text() {
    return "usage: frinfer-tokenize <model.ninfer> "
           "(--text <string> | --file <file> | --stdin)\n"
           "       [(--ids <id list> | --ids-file <file>) [--decode-text]]\n"
           "       [--pieces] [--count] [--ids-only] [--skip-special-tokens] [--device N]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n"
           "       [--version]\n"
           "\n"
           "Encodes text to token ids, or decodes ids back to text, on the artifact's own\n"
           "tokenizer. No chat template and no implicit special token is applied, so a count here\n"
           "is the raw text count and not a request's framed count.\n"
           "--pieces adds one 'id -> piece' line per token, --count prints the token count to\n"
           "stderr, and --ids-only makes stdout a bare Python-parseable id list.\n"
           "--decode-text is the inverse direction: give ids, receive text on stdout. Ids may be\n"
           "separated by whitespace, commas, or newlines, so a copy of --pieces output parses.\n"
           "--skip-special-tokens suppresses control-token spellings when decoding; the spelling\n"
           "of a special token is otherwise shown verbatim so framing is auditable.\n"
           "Startup milestones and diagnostics go to stderr; only the result goes to stdout.\n";
}

std::string require_value(int argc, char** argv, int& index, std::string_view flag) {
    if (++index >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
    return argv[index];
}

std::string read_file(const std::filesystem::path& path, std::string_view what) {
    const std::string label(what);
    std::error_code error;
    if (std::filesystem::is_directory(path, error)) {
        throw std::invalid_argument(label + " is a directory: " + path.string());
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::invalid_argument(label + " cannot open file: " + path.string()); }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (input.bad()) {
        throw std::invalid_argument(label + " failed to read file: " + path.string());
    }
    return buffer.str();
}

std::string read_stdin() {
    std::ostringstream buffer;
    buffer << std::cin.rdbuf();
    if (std::cin.bad()) { throw std::invalid_argument("--stdin failed to read stdin"); }
    return buffer.str();
}

// Ids arrive separated by whitespace, commas or newlines, so a copy of --pieces output and a
// hand-written list both parse. A malformed entry is rejected by name rather than skipped.
std::vector<ninfer::TokenId> parse_ids(std::string_view text) {
    std::vector<ninfer::TokenId> ids;
    std::size_t position = 0;
    while (position < text.size()) {
        while (position < text.size() &&
               (text[position] == ' ' || text[position] == '\t' || text[position] == '\n' ||
                text[position] == '\r' || text[position] == ',')) {
            ++position;
        }
        if (position >= text.size()) { break; }
        const std::size_t start = position;
        if (text[position] == '-' || text[position] == '+') { ++position; }
        while (position < text.size() && text[position] >= '0' && text[position] <= '9') {
            ++position;
        }
        if (position == start || (position - start == 1 && text[start] == '-')) {
            throw std::invalid_argument("token id list contains a non-numeric entry at offset " +
                                        std::to_string(start));
        }
        int value             = 0;
        const char* first     = text.data() + start;
        const char* last      = text.data() + position;
        const auto parsed     = std::from_chars(first, last, value);
        if (parsed.ec != std::errc{} || parsed.ptr != last) {
            throw std::invalid_argument("token id out of range: " + std::string(first, last));
        }
        ids.push_back(static_cast<ninfer::TokenId>(value));
    }
    return ids;
}

Options parse_options(int argc, char** argv) {
    Options options;
    bool text_set = false;
    bool ids_set  = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            options.help_requested = true;
        } else if (arg == "--version") {
            options.version_requested = true;
        } else if (arg == "--text") {
            if (text_set) { throw std::invalid_argument("pass exactly one text input"); }
            text_set     = true;
            options.text = require_value(argc, argv, i, "--text");
        } else if (arg == "--file") {
            if (text_set) { throw std::invalid_argument("pass exactly one text input"); }
            text_set     = true;
            options.file = require_value(argc, argv, i, "--file");
        } else if (arg == "--stdin") {
            if (text_set) { throw std::invalid_argument("pass exactly one text input"); }
            text_set             = true;
            options.stdin_source = true;
        } else if (arg == "--ids" || arg == "--ids-file") {
            if (ids_set) { throw std::invalid_argument("pass exactly one id input"); }
            ids_set                = true;
            const std::string value = require_value(argc, argv, i, arg);
            if (arg == "--ids") {
                options.ids = value;
            } else {
                options.ids_file = value;
            }
        } else if (arg == "--decode-text") {
            options.decode_text = true;
        } else if (arg == "--pieces") {
            options.show_pieces = true;
        } else if (arg == "--count") {
            options.show_count = true;
        } else if (arg == "--ids-only") {
            options.ids_only = true;
        } else if (arg == "--skip-special-tokens") {
            options.skip_special_tokens = true;
        } else if (arg == "--device") {
            options.device = std::stoi(require_value(argc, argv, i, "--device"));
        } else if (arg == "--log-level") {
            const std::string level = require_value(argc, argv, i, "--log-level");
            if (level == "trace") {
                options.log_level = LogLevel::Trace;
            } else if (level == "debug") {
                options.log_level = LogLevel::Debug;
            } else if (level == "info") {
                options.log_level = LogLevel::Info;
            } else if (level == "warning") {
                options.log_level = LogLevel::Warning;
            } else if (level == "error") {
                options.log_level = LogLevel::Error;
            } else if (level == "critical") {
                options.log_level = LogLevel::Critical;
            } else if (level == "off") {
                options.log_level = LogLevel::Off;
            } else {
                throw std::invalid_argument("invalid --log-level: " + level);
            }
        } else if (!arg.empty() && arg.front() == '-') {
            throw std::invalid_argument("unknown option: " + std::string(arg));
        } else if (options.artifact.empty()) {
            options.artifact = arg;
        } else {
            throw std::invalid_argument("unexpected positional argument: " + std::string(arg));
        }
    }
    if (options.help_requested || options.version_requested) { return options; }

    if (options.artifact.empty()) { throw std::invalid_argument("<model.ninfer> is required"); }
    if (options.decode_text && !ids_set) {
        throw std::invalid_argument("--decode-text needs --ids or --ids-file");
    }
    if (!options.decode_text && !text_set) {
        throw std::invalid_argument(
            "pass exactly one text input (--text, --file or --stdin), or --decode-text with ids");
    }
    if (options.artifact.extension() != ".ninfer") {
        throw std::invalid_argument("model artifact must be a .ninfer file");
    }
    options.mode = options.decode_text ? Mode::Decode : Mode::Encode;
    return options;
}

// Render a token's spelling so a token boundary is visible in the output rather than invisible in
// the terminal: a newline or a control byte must not break the one-line-per-token shape.
std::string quoted_piece(std::string_view text, bool special) {
    std::string out = "'";
    for (const char raw : text) {
        const unsigned char c = static_cast<unsigned char>(raw);
        switch (raw) {
        case '\n': out += "\\n"; continue;
        case '\r': out += "\\r"; continue;
        case '\t': out += "\\t"; continue;
        case '\\': out += "\\\\"; continue;
        case '\'': out += "\\'"; continue;
        default: break;
        }
        if (c < 0x20 || c == 0x7f) {
            static const char* kHex = "0123456789abcdef";
            out += "\\x";
            out += kHex[c >> 4U];
            out += kHex[c & 0xfU];
            continue;
        }
        out += raw;
    }
    out += '\'';
    if (special) { out += " [special]"; }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << "\n\n" << usage_text();
        return 2;
    }
    if (options.help_requested) {
        std::cout << usage_text();
        return 0;
    }
    if (options.version_requested) {
        std::cout << ninfer::product::version_text("frinfer-tokenize") << '\n';
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);

    try {
        // Ids are parsed before the Engine exists: a malformed id list is an option error, and
        // reporting it must not cost a weight load.
        std::vector<ninfer::TokenId> decode_ids;
        if (options.mode == Mode::Decode) {
            decode_ids = options.ids_file.empty()
                             ? parse_ids(options.ids)
                             : parse_ids(read_file(options.ids_file, "--ids-file"));
        }
        const std::string text = options.mode == Mode::Decode
                                     ? std::string{}
                                     : (options.stdin_source
                                            ? read_stdin()
                                            : (options.file.empty() ? options.text
                                                                    : read_file(options.file,
                                                                                "--file")));

        // Tokenizing needs no KV capacity and no workspace, so the load asks for the smallest
        // context the Engine accepts rather than a usable generation context.
        ninfer::EngineOptions engine_options;
        engine_options.artifact_path    = options.artifact;
        engine_options.purpose          = ninfer::EnginePurpose::CausalScoring;
        engine_options.device           = options.device;
        engine_options.max_context      = 2048;
        engine_options.kv_capacity      = ninfer::KvCapacityPolicy::explicit_capacity(2048);
        engine_options.use_cuda_graph   = false;
        engine_options.startup_observer = startup_log.observer();
        ninfer::Engine engine(std::move(engine_options));
        startup_log.engine_ready(engine.load_summary());

        if (options.mode == Mode::Decode) {
            // token_piece validates each id, so an out-of-vocabulary id is reported with its own id
            // rather than as a decode failure over the whole list.
            std::string decoded;
            decoded.reserve(decode_ids.size() * 4);
            for (const ninfer::TokenId id : decode_ids) {
                const ninfer::TokenPiece piece = engine.token_piece(id);
                if (!options.skip_special_tokens || !piece.special) { decoded += piece.text; }
            }
            std::cout << decoded;
            if (options.show_count) {
                std::cerr << "decoded " << decode_ids.size() << " tokens into "
                          << decoded.size() << " bytes\n";
            }
            return 0;
        }

        const std::vector<ninfer::TokenId> ids = engine.tokenize_text(text);
        if (options.show_pieces) {
            for (const ninfer::TokenId id : ids) {
                const ninfer::TokenPiece piece = engine.token_piece(id);
                std::cout << id << " -> " << quoted_piece(piece.text, piece.special) << '\n';
            }
        } else if (options.ids_only) {
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (i != 0) { std::cout << ' '; }
                std::cout << ids[i];
            }
            if (!ids.empty()) { std::cout << '\n'; }
        } else {
            std::cout << engine.detokenize(ids) << '\n';
        }
        if (options.show_count) { std::cerr << "tokens: " << ids.size() << '\n'; }
        return 0;
    } catch (const std::exception& error) {
        logger->error("{}", ninfer::product::format_pretty_text(error.what()));
        return 1;
    }
}