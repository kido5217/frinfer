// Grammar-constrained generation through the real engine route: the mask transport, the
// per-request producer, the region gate and the grammar-advance path against a real artifact.
//
// NINFER_TEST_ARTIFACT selects the artifact; without it the test skips (exit 77).
//
// The fixture runs the non-speculative backend with greedy sampling, so the constrained runs are
// deterministic: a grammar that admits exactly "9999" must force that text, a four-digit grammar
// must return four digits and stop through the grammar's own end (the mask allows the
// end-of-generation ids only where the grammar can end), and the unconstrained control on the same
// prompt is printed for comparison.

#include "ninfer/engine.h"

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

ninfer::EngineOptions constrained_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 512;
    options.kv_capacity                          = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk                        = 256;
    options.speculative.backend                  = ninfer::SpeculativeBackend::None;
    options.max_concurrency                      = 1;
    options.max_pending_requests                 = 1;
    options.context_cache.device_state_slots     = 1;
    options.context_cache.host_state_slots       = 2;
    options.context_cache.host_kv_capacity_bytes = 256ULL << 20;
    return options;
}

ninfer::RequestOptions greedy_request(std::uint32_t max_tokens,
                                      std::string grammar = {}) {
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = max_tokens;
    request.execution.sampling.temperature    = 0.0F;
    if (!grammar.empty()) {
        request.constraint.emplace(ninfer::GrammarConstraint{std::move(grammar)});
    }
    return request;
}

bool is_exact_digits(const std::string& text, std::size_t count) {
    if (text.size() != count) { return false; }
    for (const char byte : text) {
        if (std::isdigit(static_cast<unsigned char>(byte)) == 0) { return false; }
    }
    return true;
}

int exercise(const char* artifact) {
    std::string forced_text;
    std::string digits_text;
    {
        ninfer::Engine engine(constrained_engine_options(artifact));
    const auto run = [&engine](std::uint32_t max_tokens, std::string grammar = {}) {
        return engine.generate(
            engine.prepare_tokens(engine.tokenize_text("Complete the sequence: ")),
            greedy_request(max_tokens, std::move(grammar)));
    };

    const ninfer::GenerationResult control = run(32);
    std::cout << "control content=\"" << control.content << "\"\n";

    const ninfer::GenerationResult forced = run(32, "root ::= \"9999\"");
    if (forced.content != "9999") {
        std::cerr << "forced grammar did not force the text: \"" << forced.content << "\"\n";
        return 1;
    }
    if (forced.finish_reason != ninfer::FinishReason::StopToken) {
        std::cerr << "the forced run did not stop through the grammar's end\n";
        return 1;
    }

    const ninfer::GenerationResult digits = run(32, "root ::= [0-9]{4}");
    if (!is_exact_digits(digits.content, 4)) {
        std::cerr << "digit grammar did not constrain the text: \"" << digits.content << "\"\n";
        return 1;
    }
    if (digits.finish_reason != ninfer::FinishReason::StopToken) {
        std::cerr << "the digit run did not stop through the grammar's end\n";
        return 1;
    }


        forced_text = forced.content;
        digits_text = digits.content;

        // An invalid grammar fails the request synchronously instead of running unconstrained.
        bool rejected = false;
        try {
            (void)run(8, "root ::= [");
        } catch (const ninfer::RequestError& error) {
            rejected = error.kind() == ninfer::RequestErrorKind::InvalidConstraint;
        }
        if (!rejected) {
            std::cerr << "an invalid grammar was not rejected\n";
            return 1;
        }
    }

    // The MTP backend masks every verify column: the same grammars must hold under speculation,
    // where the accepted prefix also has to be fed back into the grammar state.
    {
        ninfer::EngineOptions options          = constrained_engine_options(artifact);
        options.speculative.backend            = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens       = 3;
        options.speculative.proposal_head      = ninfer::ProposalHead::Optimized;
        ninfer::Engine mtp_engine(std::move(options));
        const auto run_mtp = [&mtp_engine](std::string grammar) {
            return mtp_engine.generate(
                mtp_engine.prepare_tokens(mtp_engine.tokenize_text("Complete the sequence: ")),
                greedy_request(32, std::move(grammar)));
        };
        const ninfer::GenerationResult forced_mtp = run_mtp("root ::= \"9999\"");
        if (forced_mtp.content != "9999") {
            std::cerr << "MTP forced grammar did not force the text: \"" << forced_mtp.content
                      << "\"\n";
            return 1;
        }
        if (forced_mtp.finish_reason != ninfer::FinishReason::StopToken) {
            std::cerr << "the MTP forced run did not stop through the grammar's end\n";
            return 1;
        }
        const ninfer::GenerationResult digits_mtp = run_mtp("root ::= [0-9]{4}");
        if (!is_exact_digits(digits_mtp.content, 4)) {
            std::cerr << "the MTP digit grammar did not constrain the text: \""
                      << digits_mtp.content << "\"\n";
            return 1;
        }
        std::cout << "mtp forced=\"" << forced_mtp.content << "\" digits=\"" << digits_mtp.content
                  << "\"\n";
    }
    std::cout << "ok constrained" << " forced=\"" << forced_text << "\" digits=\"" << digits_text
              << "\"\n";
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    try {
        return exercise(artifact);
    } catch (const std::exception& error) {
        std::cerr << "constrained exercise failed: " << error.what() << '\n';
        return 1;
    }
}
