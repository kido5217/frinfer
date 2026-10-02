#pragma once

// Physical transport for the grammar mask table: the host staging mirror, the row-table geometry
// and the per-round upload / active-flag lifecycle. Grammar semantics (which tokens a lane may
// emit) stay with the caller; this unit only moves rows. The device table and flag live in the
// RoundState mask regions (round_buffers.h), whose layout this header's geometry describes.

#include "models/qwen3_5/program/round_buffers.h"

#include "core/tensor.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5 {

class MaskTransport {
public:
    // Mask rows mirror the decode batch layout: column-major over (kColumnCapacity columns,
    // kMaximumConcurrency batches), so a decode lane (column c, batch b) addresses row
    // c * kMaximumConcurrency + b. The column capacity covers the widest decode layout any
    // backend can present (DFlash target verify); MTP and ordinary layouts are prefixes of it.
    static constexpr std::uint32_t kColumnCapacity = kDFlashDecodeMaximumWidth;
    static constexpr std::uint32_t kRowCapacity    = kColumnCapacity * kMaximumConcurrency;

    static constexpr std::uint32_t lane_stride() { return kMaximumConcurrency; }

    static constexpr std::uint32_t row_index(std::uint32_t column, std::uint32_t batch) {
        return column * kMaximumConcurrency + batch;
    }

    // Rows needed to carry `batch` lanes over `columns` columns of the table.
    static constexpr std::uint32_t required_rows(std::uint32_t columns, std::uint32_t batch) {
        return columns == 0 || batch == 0 ? 0 : (columns - 1) * kMaximumConcurrency + batch;
    }

    MaskTransport() = default;
    // Host-only staging, for tests and inspection: the device side stays unattached, so upload
    // and deactivate are inert while the staging verbs work against host memory.
    explicit MaskTransport(std::size_t words);

    // Binds the persistent device table and active flag, and sizes the host staging from the
    // table's word count. A null table leaves the transport device-disabled.
    void attach(const Tensor& rows, const Tensor& active, cudaStream_t stream);

    [[nodiscard]] bool enabled() const noexcept { return rows_.data != nullptr; }

    [[nodiscard]] std::size_t words() const noexcept { return words_; }

    // Clears the [columns x row_count] staging window to all-ones: every row a round uploads is
    // rewritten, so a row left masked by an earlier round can never leak into this one.
    void begin_round(std::uint32_t columns, std::size_t row_count);
    // Contiguous [columns x words] block the grammar fills for one batch position.
    std::span<std::uint32_t> fill_block(std::uint32_t columns);
    // Scatters the fill block of `batch_position` into the staging table at its column rows.
    void publish(std::uint32_t batch_position, std::uint32_t columns);
    // One upload per column (within a column the used rows are contiguous, across columns they are
    // kMaximumConcurrency rows apart) and raises the active flag when clear.
    void upload(std::uint32_t columns, std::size_t row_count);
    // Lowers the active flag when raised.
    void deactivate();

    // One staged table row: the bytes `upload` transfers for that lane.
    [[nodiscard]] std::span<const std::uint32_t> staged_row(std::uint32_t column,
                                                            std::uint32_t batch) const;

private:
    [[nodiscard]] bool has_staging() const noexcept { return words_ != 0; }

    Tensor rows_;
    Tensor active_;
    cudaStream_t stream_ = nullptr;
    // [words, kRowCapacity] mirror of the device table, and the [words, kColumnCapacity]
    // per-position block the grammar fills before the scatter into the mirror.
    std::vector<std::uint32_t> staging_;
    std::vector<std::uint32_t> fill_;
    std::size_t words_        = 0;
    std::int32_t active_host_ = 0;
};

} // namespace ninfer::models::qwen3_5
