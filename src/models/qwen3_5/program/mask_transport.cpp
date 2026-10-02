#include "models/qwen3_5/program/mask_transport.h"

#include "core/device.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5 {

MaskTransport::MaskTransport(std::size_t words) {
    if (words == 0) { return; }
    words_ = words;
    staging_.assign(words_ * kRowCapacity, 0xFFFFFFFFU);
    fill_.assign(words_ * kColumnCapacity, 0xFFFFFFFFU);
}

void MaskTransport::attach(const Tensor& rows, const Tensor& active, cudaStream_t stream) {
    if (rows.data == nullptr) {
        rows_        = Tensor{};
        active_      = Tensor{};
        stream_      = nullptr;
        words_       = 0;
        active_host_ = 0;
        staging_.clear();
        fill_.clear();
        return;
    }
    if (rows.ne[1] < static_cast<std::int32_t>(kRowCapacity)) {
        throw std::invalid_argument("mask transport table is shorter than the row capacity");
    }
    rows_   = rows;
    active_ = active;
    stream_ = stream;
    words_  = static_cast<std::size_t>(rows.ne[0]);
    staging_.assign(words_ * kRowCapacity, 0xFFFFFFFFU);
    fill_.assign(words_ * kColumnCapacity, 0xFFFFFFFFU);
    // The program memsets the device flag to zero before attach; mirror that state.
    active_host_ = 0;
}

void MaskTransport::begin_round(std::uint32_t columns, std::size_t row_count) {
    if (!has_staging() || columns == 0 || row_count == 0) { return; }
    for (std::uint32_t column = 0; column < columns; ++column) {
        for (std::size_t position = 0; position < row_count; ++position) {
            const std::size_t row =
                static_cast<std::size_t>(row_index(column, static_cast<std::uint32_t>(position)));
            std::fill_n(staging_.data() + row * words_, words_, 0xFFFFFFFFU);
        }
    }
}

std::span<std::uint32_t> MaskTransport::fill_block(std::uint32_t columns) {
    if (!has_staging() || columns == 0) { return {}; }
    return {fill_.data(), static_cast<std::size_t>(columns) * words_};
}

void MaskTransport::publish(std::uint32_t batch_position, std::uint32_t columns) {
    if (!has_staging() || columns == 0) { return; }
    for (std::uint32_t column = 0; column < columns; ++column) {
        const std::size_t source = static_cast<std::size_t>(column) * words_;
        const std::size_t target =
            static_cast<std::size_t>(row_index(column, batch_position)) * words_;
        std::copy_n(fill_.data() + source, words_, staging_.data() + target);
    }
}

void MaskTransport::upload(std::uint32_t columns, std::size_t row_count) {
    if (!enabled() || columns == 0 || row_count == 0) { return; }
    for (std::uint32_t column = 0; column < columns; ++column) {
        const std::size_t row_offset = static_cast<std::size_t>(row_index(column, 0)) * words_;
        CUDA_CHECK(cudaMemcpyAsync(
            static_cast<unsigned char*>(rows_.data) + row_offset * sizeof(std::uint32_t),
            staging_.data() + row_offset, row_count * words_ * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice, stream_));
    }
    if (active_host_ != 1) {
        const std::int32_t active = 1;
        CUDA_CHECK(cudaMemcpyAsync(active_.data, &active, sizeof(active), cudaMemcpyHostToDevice,
                                   stream_));
        active_host_ = 1;
    }
}

void MaskTransport::deactivate() {
    if (!enabled() || active_host_ == 0) { return; }
    const std::int32_t inactive = 0;
    CUDA_CHECK(cudaMemcpyAsync(active_.data, &inactive, sizeof(inactive), cudaMemcpyHostToDevice,
                               stream_));
    active_host_ = 0;
}

std::span<const std::uint32_t> MaskTransport::staged_row(std::uint32_t column,
                                                         std::uint32_t batch) const {
    if (!has_staging() || column >= kColumnCapacity || batch >= kMaximumConcurrency) { return {}; }
    return {staging_.data() + static_cast<std::size_t>(row_index(column, batch)) * words_, words_};
}

} // namespace ninfer::models::qwen3_5
