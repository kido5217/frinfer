#include "ninfer/ops/apply_mask.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::uint16_t kMinusInfBits = 0xFF80u;

std::uint32_t next_random(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

// Arbitrary BF16 payloads with the -inf bit pattern excluded, so expected writes are unambiguous.
std::vector<std::uint16_t> make_logits(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(count);
    std::uint32_t state = seed;
    for (auto& value : bits) {
        const std::uint32_t random = next_random(state);
        value                      = static_cast<std::uint16_t>(random >> 8);
        if (value == kMinusInfBits) { value = 0u; }
    }
    return bits;
}

std::vector<std::uint32_t> make_mask_rows(std::int32_t rows, std::int32_t words,
                                          std::int32_t token_domain, bool all_ones,
                                          std::uint32_t seed) {
    std::vector<std::uint32_t> words_buffer(static_cast<std::size_t>(rows) * words);
    std::uint32_t state = seed;
    for (auto& word : words_buffer) { word = all_ones ? 0xFFFFFFFFu : next_random(state); }
    if (!all_ones) {
        // Padding bits beyond the token domain must be ignored: clear them everywhere.
        const std::int32_t tail = token_domain % 32;
        if (tail != 0) {
            const std::uint32_t keep = (1u << tail) - 1u;
            for (std::int32_t row = 0; row < rows; ++row) {
                words_buffer[static_cast<std::size_t>(row) * words + (words - 1)] &= keep;
            }
        }
    }
    return words_buffer;
}

bool mask_bit(const std::vector<std::uint32_t>& masks, std::int32_t row, std::int32_t words,
              std::int32_t token) {
    return (masks[static_cast<std::size_t>(row) * words + (token >> 5)] >> (token & 31)) & 1u;
}

int run_2d_case(const char* label, std::int32_t vocab, std::int32_t token_domain,
                std::int32_t batch, const std::vector<std::uint32_t>& masks, bool active_flag,
                std::uint32_t seed) {
    const std::int32_t words = (token_domain + 31) / 32;
    const std::int32_t rows  = 8;
    const std::vector<std::uint16_t> input =
        make_logits(static_cast<std::size_t>(vocab) * static_cast<std::size_t>(batch), seed);

    GuardedDeviceBuffer logits_buffer(input.size() * sizeof(std::uint16_t));
    logits_buffer.copy_from_host(input.data(), input.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer masks_buffer(masks.size() * sizeof(std::uint32_t));
    masks_buffer.copy_from_host(masks.data(), masks.size() * sizeof(std::uint32_t));
    const std::int32_t active_host = active_flag ? 1 : 0;
    GuardedDeviceBuffer active_buffer(sizeof(std::int32_t));
    active_buffer.copy_from_host(&active_host, sizeof(active_host));

    Tensor logits(logits_buffer.data(), DType::BF16, {vocab, batch});
    Tensor mask_tensor(masks_buffer.data(), DType::I32, {words, rows});
    Tensor active_tensor(active_buffer.data(), DType::I32, {1});
    ops::apply_mask(logits, /*columns=*/1, batch, mask_tensor, active_tensor, token_domain,
                    /*mask_lane_stride=*/8, nullptr);
    cuda_synchronize();

    std::vector<std::uint16_t> expected = input;
    if (active_flag) {
        for (std::int32_t row = 0; row < batch; ++row) {
            for (std::int32_t token = 0; token < token_domain; ++token) {
                if (!mask_bit(masks, row, words, token)) {
                    expected[static_cast<std::size_t>(token) +
                             static_cast<std::size_t>(row) * vocab] = kMinusInfBits;
                }
            }
        }
    }
    int failures = verify_exact(
        label, from_device<std::uint16_t>(logits_buffer.data(), input.size()), expected);
    failures += logits_buffer.verify_guards("apply_mask logits");
    failures += masks_buffer.verify_guards("apply_mask masks");
    failures += active_buffer.verify_guards("apply_mask active");
    return failures;
}

int run_3d_case(const char* label, std::uint32_t seed) {
    constexpr std::int32_t vocab     = 129;
    constexpr std::int32_t domain    = 125;
    constexpr std::int32_t words     = (domain + 31) / 32;
    constexpr std::int32_t columns   = 6;
    constexpr std::int32_t batch_cap = 4;
    constexpr std::int32_t batch     = 2;
    constexpr std::int32_t rows      = 48;
    const std::vector<std::uint16_t> input =
        make_logits(static_cast<std::size_t>(vocab) * columns * batch_cap, seed);
    const std::vector<std::uint32_t> masks = make_mask_rows(rows, words, domain, false, seed + 3);

    GuardedDeviceBuffer logits_buffer(input.size() * sizeof(std::uint16_t));
    logits_buffer.copy_from_host(input.data(), input.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer masks_buffer(masks.size() * sizeof(std::uint32_t));
    masks_buffer.copy_from_host(masks.data(), masks.size() * sizeof(std::uint32_t));
    const std::int32_t active_host = 1;
    GuardedDeviceBuffer active_buffer(sizeof(std::int32_t));
    active_buffer.copy_from_host(&active_host, sizeof(active_host));

    Tensor region(logits_buffer.data(), DType::BF16, {vocab, columns, batch_cap});
    Tensor logits = region.slice(2, 0, batch);
    Tensor mask_tensor(masks_buffer.data(), DType::I32, {words, rows});
    Tensor active_tensor(active_buffer.data(), DType::I32, {1});
    ops::apply_mask(logits, columns, batch, mask_tensor, active_tensor, domain,
                    /*mask_lane_stride=*/8, nullptr);
    cuda_synchronize();

    // Only the sliced batch lanes may change; the untouched capacity lanes stay byte-identical.
    std::vector<std::uint16_t> expected = input;
    for (std::int32_t column = 0; column < columns; ++column) {
        for (std::int32_t row = 0; row < batch; ++row) {
            const std::int32_t mask_row = column * 8 + row;
            for (std::int32_t token = 0; token < domain; ++token) {
                if (!mask_bit(masks, mask_row, words, token)) {
                    const std::size_t index = static_cast<std::size_t>(token) +
                                              static_cast<std::size_t>(column) * vocab +
                                              static_cast<std::size_t>(row) * vocab * columns;
                    expected[index] = kMinusInfBits;
                }
            }
        }
    }
    int failures = verify_exact(
        label, from_device<std::uint16_t>(logits_buffer.data(), input.size()), expected);
    // The batch rows outside the slice must remain untouched even though masks cover them.
    for (std::int32_t row = batch; row < batch_cap; ++row) {
        for (std::int32_t token = 0; token < vocab; ++token) {
            for (std::int32_t column = 0; column < columns; ++column) {
                const std::size_t index = static_cast<std::size_t>(token) +
                                          static_cast<std::size_t>(column) * vocab +
                                          static_cast<std::size_t>(row) * vocab * columns;
                if (expected[index] != input[index]) {
                    std::cerr << label << ": unsliced lane changed at token " << token << '\n';
                    ++failures;
                }
            }
        }
    }
    failures += logits_buffer.verify_guards("apply_mask 3d logits");
    failures += masks_buffer.verify_guards("apply_mask 3d masks");
    failures += active_buffer.verify_guards("apply_mask 3d active");
    return failures;
}

int run_validation_case(const char* label, const std::function<void()>& body) {
    try {
        body();
    } catch (const std::exception&) { return 0; }
    std::cerr << label << ": expected an exception\n";
    return 1;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    constexpr std::int32_t vocab  = 257;
    constexpr std::int32_t domain = 250;
    constexpr std::int32_t batch  = 3;
    constexpr std::int32_t words  = (domain + 31) / 32;
    constexpr std::int32_t rows   = 8;

    int failures = 0;
    failures += run_2d_case("apply_mask 2d exact", vocab, domain, batch,
                            make_mask_rows(rows, words, domain, false, 0x1234u), true, 0x51u);
    failures += run_2d_case("apply_mask 2d inactive", vocab, domain, batch,
                            make_mask_rows(rows, words, domain, false, 0x1234u), false, 0x52u);
    failures += run_2d_case("apply_mask 2d all-ones", vocab, domain, batch,
                            make_mask_rows(rows, words, domain, true, 0x1234u), true, 0x53u);
    failures += run_3d_case("apply_mask 3d strided", 0x61u);

    failures += run_validation_case("apply_mask rejects non-BF16 logits", [] {
        GuardedDeviceBuffer logits_buffer(16 * 2);
        GuardedDeviceBuffer masks_buffer(8 * 4);
        GuardedDeviceBuffer active_buffer(4);
        Tensor logits(logits_buffer.data(), DType::FP32, {16, 1});
        Tensor masks(masks_buffer.data(), DType::I32, {8, 1});
        Tensor active(active_buffer.data(), DType::I32, {1});
        ops::apply_mask(logits, 1, 1, masks, active, 16, 8, nullptr);
    });
    failures += run_validation_case("apply_mask rejects a short mask table", [] {
        GuardedDeviceBuffer logits_buffer(16 * 2 * 2);
        GuardedDeviceBuffer masks_buffer(4);
        GuardedDeviceBuffer active_buffer(4);
        Tensor logits(logits_buffer.data(), DType::BF16, {16, 2});
        Tensor masks(masks_buffer.data(), DType::I32, {1, 1});
        Tensor active(active_buffer.data(), DType::I32, {1});
        ops::apply_mask(logits, 1, 2, masks, active, 16, 8, nullptr);
    });
    failures += run_validation_case("apply_mask rejects an oversized domain", [] {
        GuardedDeviceBuffer logits_buffer(16 * 2);
        GuardedDeviceBuffer masks_buffer(8 * 4 * 2);
        GuardedDeviceBuffer active_buffer(4);
        Tensor logits(logits_buffer.data(), DType::BF16, {16, 1});
        Tensor masks(masks_buffer.data(), DType::I32, {8, 2});
        Tensor active(active_buffer.data(), DType::I32, {1});
        ops::apply_mask(logits, 1, 1, masks, active, 33, 8, nullptr);
    });

    std::cout << (failures ? "FAIL" : "OK") << " apply_mask\n";
    return failures ? 1 : 0;
}
