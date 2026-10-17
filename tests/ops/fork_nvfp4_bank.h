#pragma once
//
// Shared device-bank fixture for the fork NVFP4 MoE route tests (#211-#213 lineage).
//
// The route consumes resident experts through an affine ExpertBankDeviceView, so the fixture
// packs each expert's gate/up pair as one 2I-row encoded parent ([codes | scales], gate rows
// first) and each down projection as one H-row parent, uploads them as stride-indexed entry
// arrays with the per-expert FP32 divisor planes, and exposes both the CudaSwiGLUExperts view
// for the launch and the host-side patterned weights for the FP64 oracle
// (quantized_weight::logical_weight_fp64).

#include "core/buffer.h"
#include "ops/moe/kernels/cuda/fork_nvfp4/launch.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cstdint>
#include <vector>

namespace ninfer::test::fork_nvfp4 {

namespace route = ninfer::ops::detail::moe_cuda::fork_nvfp4;
namespace qw    = ninfer::test::quantized_weight;

inline constexpr std::int32_t kHidden       = route::kHidden;
inline constexpr std::int32_t kExperts      = route::kExperts;
inline constexpr std::int32_t kTopK         = route::kTopK;
inline constexpr std::int32_t kIntermediate = route::kIntermediate;

inline constexpr std::uint64_t kGuCodes  = std::uint64_t(2 * kIntermediate) * kHidden / 2;
inline constexpr std::uint64_t kGuEntry  = kGuCodes + std::uint64_t(2 * kIntermediate) *
                                                              (kHidden / 16);
inline constexpr std::uint64_t kDnCodes  = std::uint64_t(kHidden) * kIntermediate / 2;
inline constexpr std::uint64_t kDnEntry  = kDnCodes + std::uint64_t(kHidden) *
                                                              (kIntermediate / 16);

inline qw::PackedWeight make_nvfp4(std::int32_t n, std::int32_t k, std::uint32_t seed,
                                   float divisor) {
    return qw::make_patterned_weight(
        QType::NVFP4, n, k, seed,
        qw::PatternedWeightOptions{qw::RowSplitScalePattern::Unit,
                                   qw::RowSplitCodePattern::Coordinate, divisor, divisor});
}

class Bank {
public:
    explicit Bank(std::uint32_t seed = 0x5EEDu) : device_(cuda_environment().context.device()) {
        gate_up_.resize(std::size_t(kExperts) * kGuEntry, 0);
        down_.resize(std::size_t(kExperts) * kDnEntry, 0);
        gu_divisors_.reserve(kExperts);
        dn_divisors_.reserve(kExperts);
        gate_.reserve(kExperts);
        up_.reserve(kExperts);
        downs_.reserve(kExperts);
        for (std::int32_t expert = 0; expert < kExperts; ++expert) {
            // gate/up share one per-expert divisor (the source's rule); down carries its own.
            const float gate_divisor = 2.0F + 0.25F * static_cast<float>(expert);
            const float down_divisor = 3.0F + 0.5F * static_cast<float>(expert);
            gate_.push_back(make_nvfp4(kIntermediate, kHidden,
                                       seed + 0x9E3779B9u + expert * 2u, gate_divisor));
            up_.push_back(make_nvfp4(kIntermediate, kHidden,
                                     seed + 0x85EBCA6Bu + expert * 2u + 1u, gate_divisor));
            downs_.push_back(make_nvfp4(kHidden, kIntermediate,
                                        seed + 0xC2B2AE3Du + expert * 3u, down_divisor));
            pack(gate_up_.data() + std::size_t(expert) * kGuEntry, gate_.back(), up_.back());
            const auto& d   = downs_.back();
            std::copy(d.payload.begin(),
                      d.payload.begin() + std::ptrdiff_t(d.scale_plane_offset),
                      down_.data() + std::size_t(expert) * kDnEntry);
            std::copy(d.payload.begin() + std::ptrdiff_t(d.scale_plane_offset),
                      d.payload.begin() +
                          std::ptrdiff_t(d.scale_plane_offset + d.scale_plane_bytes),
                      down_.data() + std::size_t(expert) * kDnEntry + kDnCodes);
            gu_divisors_.push_back(gate_divisor);
            dn_divisors_.push_back(down_divisor);
        }
        gu_device_            = {device_, gate_up_.size()};
        dn_device_            = {device_, down_.size()};
        gu_divisors_device_   = {device_, kExperts * sizeof(float)};
        dn_divisors_device_   = {device_, kExperts * sizeof(float)};
        gu_device_.copy_from_host(gate_up_.data(), gate_up_.size());
        dn_device_.copy_from_host(down_.data(), down_.size());
        gu_divisors_device_.copy_from_host(gu_divisors_.data(), kExperts * 4);
        dn_divisors_device_.copy_from_host(dn_divisors_.data(), kExperts * 4);
    }

    [[nodiscard]] ninfer::ops::CudaSwiGLUExperts experts() const {
        return {view(), {}};
    }

    [[nodiscard]] ninfer::ExpertBankDeviceView view() const {
        ninfer::ExpertBankDeviceView bank;
        bank.gate_up = matrix(gu_device_.span(), gu_divisors_device_.span(), kGuEntry, kGuCodes);
        bank.down = matrix(dn_device_.span(), dn_divisors_device_.span(), kDnEntry, kDnCodes);
        bank.count  = kExperts;
        bank.device = device_;
        return bank;
    }

    // FP64 oracle decode of the logical parent elements.
    [[nodiscard]] double gate_weight(int expert, std::int32_t row, std::int32_t column) const {
        return qw::logical_weight_fp64(gate_[expert], row, column);
    }
    [[nodiscard]] double up_weight(int expert, std::int32_t row, std::int32_t column) const {
        return qw::logical_weight_fp64(up_[expert], row, column);
    }
    [[nodiscard]] double down_weight(int expert, std::int32_t row, std::int32_t column) const {
        return qw::logical_weight_fp64(downs_[expert], row, column);
    }

    // Raw packed parents for bulk independent decode in the oracles.
    [[nodiscard]] const qw::PackedWeight& gate_packed(int expert) const { return gate_[expert]; }
    [[nodiscard]] const qw::PackedWeight& up_packed(int expert) const { return up_[expert]; }
    [[nodiscard]] const qw::PackedWeight& down_packed(int expert) const { return downs_[expert]; }

    // Frees the gate/up host payloads and the host entry image once the oracle no longer
    // decodes them (the grouped-route oracle only needs the down parents and read-back planes).
    void release_gate_up() {
        for (auto& parent : gate_) { parent.payload.clear(); parent.payload.shrink_to_fit(); }
        for (auto& parent : up_) { parent.payload.clear(); parent.payload.shrink_to_fit(); }
        gate_up_.clear();
        gate_up_.shrink_to_fit();
    }

private:
    static void pack(std::uint8_t* destination, const qw::PackedWeight& gate,
                     const qw::PackedWeight& up) {
        const auto plane = [](const qw::PackedWeight& w) {
            return std::make_pair(w.payload.data(), w.payload.data() + w.scale_plane_offset);
        };
        const auto [gate_codes, gate_scales] = plane(gate);
        const auto [up_codes, up_scales]     = plane(up);
        const std::size_t code_bytes         = kGuCodes / 2, scale_bytes = kGuCodes / 2 / 8;
        std::copy(gate_codes, gate_codes + code_bytes, destination);
        std::copy(up_codes, up_codes + code_bytes, destination + code_bytes);
        std::copy(gate_scales, gate_scales + scale_bytes, destination + kGuCodes);
        std::copy(up_scales, up_scales + scale_bytes, destination + kGuCodes + scale_bytes);
    }

    static ninfer::ExpertMatrixDeviceView matrix(ConstMemorySpan storage, ConstMemorySpan divisors,
                                                 std::uint64_t entry, std::uint64_t code_bytes) {
        ninfer::ExpertMatrixDeviceView out;
        out.codes          = static_cast<const std::byte*>(storage.data);
        out.scales         = static_cast<const std::byte*>(storage.data) + code_bytes;
        out.code_stride    = entry;
        out.scale_stride   = entry;
        out.divisors       = static_cast<const float*>(divisors.data);
        out.divisor_stride = 1;
        return out;
    }

    Device device_;
    std::vector<std::uint8_t> gate_up_, down_;
    std::vector<float> gu_divisors_, dn_divisors_;
    std::vector<qw::PackedWeight> gate_, up_, downs_;
    DeviceBuffer gu_device_, dn_device_, gu_divisors_device_, dn_divisors_device_;
};

// Route construction: distinct-per-token expert ids (29 is coprime with 256) and coefficients
// consumed without renormalisation.
inline void fill_routes(std::vector<int>& ids, std::vector<float>& coefficients, int tokens,
                        int expert_spread = 1) {
    ids.assign(std::size_t(tokens) * kTopK, 0);
    coefficients.assign(std::size_t(tokens) * kTopK, 0.0F);
    for (int token = 0; token < tokens; ++token) {
        for (int rank = 0; rank < kTopK; ++rank) {
            ids[token * kTopK + rank] =
                (token + rank * 29 + token * expert_spread * 5) % kExperts;
            coefficients[token * kTopK + rank] = 0.5F / static_cast<float>(kTopK + rank);
        }
    }
}

} // namespace ninfer::test::fork_nvfp4
