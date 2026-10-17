// Host-only coverage for the nvfp4 routed-bank admission and route-plan path. The retired
// prepare_sparse_moe_weights validator was replaced by the merged surface: routed_swiglu_profile
// checks bank geometry, the workspace query applies the Op's codec admission, the shape policy
// resolves the fork route per token count, and ExpertInputUsage enforces the source-divisor rule.
// No artifact and no CUDA device are required: this exercises the accept/reject logic that the
// real-artifact loading test cannot cover without --artifact.

#include "core/weight.h"
#include "core/weight_view.h"
#include "ninfer/ops/moe.h"
#include "ninfer/ops/weight_input.h"
#include "ops/moe/routed_swiglu/cuda/dispatch.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::ops;

// Exact fork 35B-A3B geometry.
constexpr std::int32_t kHidden       = 2048;
constexpr std::int32_t kExperts      = 256;
constexpr std::int32_t kTopK         = 8;
constexpr std::int32_t kIntermediate = 512;
constexpr std::int32_t kGateRows     = kIntermediate;

int g_failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++g_failures;
    }
}

template <class Fn>
void expect_invalid(Fn&& fn, const char* message) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << message << " (wrong error: " << error.what() << ")\n";
        ++g_failures;
        return;
    }
    std::cerr << "FAIL: " << message << " (no error)\n";
    ++g_failures;
}

// Bank layout numbers only; the profile and workspace-query paths never dereference the parent,
// so the regions describe the full 2I-row-per-expert and H-row-per-expert banks the loader binds.
struct BankLayouts {
    WeightBankLayout gate_up;
    WeightBankLayout down;
};

BankLayouts layouts(QType down_format = QType::NVFP4,
                    QuantLayout gate_layout = QuantLayout::BlockScaleK16M128x4) {
    const std::uint64_t gu_code_bytes = std::uint64_t(2 * kIntermediate) * kHidden / 2;
    const std::uint64_t gu_scale_bytes =
        std::uint64_t(2 * kIntermediate) * (kHidden / 16);
    const std::uint64_t dn_code_bytes = std::uint64_t(kHidden) * kIntermediate / 2;
    const std::uint64_t dn_scale_bytes =
        std::uint64_t(kHidden) * (kIntermediate / 16);
    BankLayouts out;
    out.gate_up = {.region =
                       {0, std::uint32_t(kExperts), std::uint64_t(2 * kIntermediate),
                        std::uint64_t(2 * kIntermediate)},
                   .format    = QType::NVFP4,
                   .layout    = gate_layout,
                   .columns   = kHidden,
                   .codes     = {0, gu_code_bytes, gu_code_bytes + gu_scale_bytes},
                   .scales    = {gu_code_bytes, gu_scale_bytes, gu_code_bytes + gu_scale_bytes},
                   .divisor_begin  = 0,
                   .divisor_stride = 1,
                   .has_divisors   = true};
    out.down = {.region =
                    {0, std::uint32_t(kExperts), std::uint64_t(kHidden),
                     std::uint64_t(kHidden)},
                .format    = down_format,
                .layout    = gate_layout,
                .columns   = kIntermediate,
                .codes     = {0, dn_code_bytes, dn_code_bytes + dn_scale_bytes},
                .scales    = {dn_code_bytes, dn_scale_bytes, dn_code_bytes + dn_scale_bytes},
                .divisor_begin  = 0,
                .divisor_stride = 1,
                .has_divisors   = true};
    return out;
}

RoutedSwiGLUProfile profile(const BankLayouts& bank, LinearPolicy policy = LinearPolicy::AllowA4) {
    return routed_swiglu_profile(bank.gate_up, bank.down, kTopK, policy, policy);
}

std::size_t capacity(const RoutedSwiGLUProfile& p, int first, int last) {
    return routed_swiglu_workspace_capacity_bytes(p, ExpertSubsetKind::All, DType::FP32,
                                                 MoeEpilogue::Write, first, last);
}

void accepts_nvfp4_route() {
    const auto p = profile(layouts());
    check(p.ffn.hidden == kHidden && p.ffn.intermediate == kIntermediate &&
              p.ffn.gate_up_type == QType::NVFP4 && p.ffn.down_type == QType::NVFP4 &&
              p.experts == kExperts && p.top_k == kTopK,
          "nvfp4 route bank was not reported with the fork geometry");
    check(capacity(p, 20, 200) > 0, "nvfp4 route workspace query returned no capacity");
    // A16-only admission (the decode/small-T front) must be accepted as well.
    check(capacity(profile(layouts(), LinearPolicy::A16Only), 1, 19) > 0,
          "nvfp4 A16-only profile was rejected");
}

void rejects_incompatible_banks() {
    {
        auto bank = layouts();
        bank.gate_up.format = QType::NVFP4;
        bank.down.format    = QType::Q4_G64_FP16;
        expect_invalid([&] { (void)capacity(profile(bank), 20, 200); },
                       "an nvfp4 bank with q4 down was accepted");
    }
    {
        auto bank = layouts(QType::NVFP4, QuantLayout::RowSplit);
        expect_invalid([&] { (void)capacity(profile(bank), 20, 200); },
                       "an nvfp4 bank with a row-split layout was accepted");
    }
    {
        auto bank = layouts();
        bank.gate_up.region.rows = 2 * kIntermediate + 1; // odd rows
        expect_invalid([&] { (void)profile(bank); },
                       "a routed gate/up bank with the wrong geometry was accepted");
    }
    {
        auto bank = layouts();
        bank.gate_up.region.count = 512;
        bank.gate_up.region.rows  = 2 * kIntermediate;
        expect_invalid([&] { (void)profile(bank); },
                       "routed banks with different expert counts were accepted");
    }
}

// Stable parent/view storage: deques keep element references valid as the bank is filled.
struct Storage {
    std::deque<std::vector<std::byte>> buffers;
    std::deque<WeightParent> parents;
    std::deque<WeightView> views;

    WeightParent& parent(QType format, QuantLayout layout, std::uint64_t rows, std::uint64_t cols) {
        const std::array<std::uint64_t, 2> shape{rows, cols};
        const auto geometry = weight_geometry(format, layout, shape);
        auto& bytes         = buffers.emplace_back(std::max<std::size_t>(geometry.bytes, 16));
        return parents.emplace_back(
            WeightParent{geometry, Device::cpu(), bytes.data(), {}});
    }

    WeightView& view(const WeightParent& parent, std::uint64_t rows, std::uint64_t cols,
                     std::uint64_t element_begin, std::uint64_t element_end) {
        return views.emplace_back(
            WeightView{{rows, cols}, {{&parent, element_begin, element_end}}});
    }
};

struct BankOptions {
    bool up_divisor_differs = false; // A4 gate/up activation divisors disagree
};

void checks_nvfp4_activation_divisors() {
    Storage storage;
    auto& gu_parent = storage.parent(QType::NVFP4, QuantLayout::BlockScaleK16M128x4,
                                     2 * kIntermediate, kHidden);
    const auto elements = std::uint64_t(2 * kIntermediate) * kHidden;
    const auto half     = elements / 2;
    const auto& gate    = storage.view(gu_parent, kGateRows, kHidden, 0, half);
    const auto& up      = storage.view(gu_parent, kGateRows, kHidden, half, elements);

    ExpertInputUsage usage;
    usage.include(WeightInput{gate, LinearPolicy::AllowA4, 1.5F});
    usage.include(
        WeightInput{up, LinearPolicy::AllowA4,
                    static_cast<float>(1.75)}); // gate/up share one source divisor
    expect_invalid([&] { (void)usage.activation_divisor(QType::NVFP4); },
                   "nvfp4 gate/up activation divisors disagreeing was accepted");

    ExpertInputUsage common;
    common.include(WeightInput{gate, LinearPolicy::AllowA4, 1.5F});
    common.include(WeightInput{up, LinearPolicy::AllowA4, 1.5F});
    check(common.activation_divisor(QType::NVFP4) == std::optional<float>(1.5F) &&
              common.policy() == LinearPolicy::AllowA4,
          "nvfp4 gate/up inputs lost their common activation divisor");
}

void checks_nvfp4_route_plan() {
    // The public workspace query and the shape policy must agree on the route boundary: they share
    // one allocation function, and the nvfp4 activation planes are part of that workspace.
    const auto p         = profile(layouts());
    const auto& dispatch = detail::moe_routed::shape_dispatch(p);
    check(std::find(dispatch.allocation_boundaries.begin(), dispatch.allocation_boundaries.end(),
                    detail::moe_routed::kernel::fork_nvfp4::kGroupedFirstTokens) !=
              dispatch.allocation_boundaries.end(),
          "the nvfp4 route does not declare its grouped-boundary allocation point");
    const auto wide = dispatch.select(200, false);
    check(wide.implementation == detail::moe_cuda::ExpertImplementation::ForkNvfp4 &&
              wide.fork.grouped,
          "nvfp4 does not select the grouped route at T>=20");
    const auto narrow = dispatch.select(19, false);
    check(narrow.implementation == detail::moe_cuda::ExpertImplementation::ForkNvfp4 &&
              !narrow.fork.grouped,
          "nvfp4 does not select the A16 route below the grouped boundary");
    // Workspace must cover both compact FP4 activation planes (input and product) plus the
    // BF16 grouped contributions at the interval top.
    const std::size_t cap          = capacity(p, 20, 200);
    const std::size_t assignments  = std::size_t(200) * kTopK;
    const std::size_t activation_plane = std::size_t(200) * (kHidden / 2 + kHidden / 16);
    const std::size_t product_plane =
        assignments * (std::size_t(kIntermediate) / 2 + kIntermediate / 16);
    const std::size_t contributions = assignments * kHidden * sizeof(std::uint16_t);
    check(cap >= activation_plane + product_plane + contributions,
          "nvfp4 grouped workspace is smaller than its activation and contribution planes");
}

} // namespace

int main() {
    try {
        accepts_nvfp4_route();
        rejects_incompatible_banks();
        checks_nvfp4_activation_divisors();
        checks_nvfp4_route_plan();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << '\n';
        return 1;
    }
    std::cout << (g_failures == 0 ? "OK" : "FAIL") << " sparse_moe nvfp4 binding\n";
    return g_failures == 0 ? 0 : 1;
}
