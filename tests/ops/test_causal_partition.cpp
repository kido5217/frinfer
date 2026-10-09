// Synthetic-SM qualification for the SM-derived causal-cache plan makers (group-(c) refactor).
//
// Every make_*_kv_causal_plan(..., multiprocessor_count) is otherwise only exercised at the host's
// own SM count, so the off-target behavior the refactor exists to provide is unprotected. These
// are pure host functions: this test needs no CUDA device, only the plan helpers themselves. The
// expectations below are the derived geometry computed independently from the documented policy
// (maximize waves-per-partition coverage, no more than 10% SM underfill for complete query-tile
// groups, cap splits at the per-family maximum), written out as literals rather than re-derived
// from the implementation.

#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"

#include <cstdint>
#include <iostream>

using namespace ninfer::ops;
using namespace ninfer::ops::detail;

namespace {

int g_failures = 0;

void expect_eq(const char* label, long long got, long long expected) {
    if (got != expected) {
        std::cerr << label << ": got " << got << " expected " << expected << '\n';
        ++g_failures;
    }
}

void expect_bool(const char* label, bool got, bool expected) {
    if (got != expected) {
        std::cerr << label << ": got " << (got ? "true" : "false") << " expected "
                  << (expected ? "true" : "false") << '\n';
        ++g_failures;
    }
}

CausalAttentionExecutionEnvelope envelop(std::uint32_t min, std::uint32_t max) {
    return {min, max};
}

// --- pure helpers over synthetic SM counts ----------------------------------

void check_underfill_helper() {
    struct Case {
        int sms;
        int independent_tiles;
        bool expected;
    };
    // filled = (sms / tiles) * tiles; underfill when filled < sms * 9 / 10.
    const Case cases[] = {
        {10, 8, true},   // filled 8 < 9
        {10, 4, true},   // filled 8 < 9
        {10, 5, false},  // filled 10 >= 9
        {8, 8, false},   // filled 8 >= 7
        {2, 3, true},    // filled 0 < 1
        {5, 3, true},    // filled 3 < 4
        {1, 1, false},   // filled 1 >= 0
        {170, 16, false},
        {170, 32, false},
        {170, 64, true},   // filled 128 < 153
        {170, 85, false},
        {170, 100, true},  // filled 100 < 153
    };
    for (const Case& c : cases) {
        expect_bool("causal_query_tiles_underfill_sms",
                    causal_query_tiles_underfill_sms(c.independent_tiles, c.sms), c.expected);
    }
}

void check_partition_target_helper() {
    struct Case {
        std::int64_t budget;
        int independent_tiles;
        int expected;
    };
    const Case cases[] = {
        {170, 1, 170},
        {340, 8, 42},
        {20, 8, 2},
        {8, 8, 1},
        {600, 4, 150},
        {1000000, 2, 256},  // clamped to CausalKvPartition::kMaxSplits
        {0, 5, 1},          // clamped up
    };
    for (const Case& c : cases) {
        expect_eq("causal_partition_target",
                  causal_partition_target(c.budget, c.independent_tiles), c.expected);
    }
}

// --- k8v4 -------------------------------------------------------------------

void check_k8v4() {
    // sms=2, heads=24, W=1 -> Grouped; decode doubles the budget (2*2=4) over 4 independent tiles.
    {
        const auto plan = make_k8v4_kv_causal_plan(24, 1, 1, envelop(1, 4096), 2);
        expect_eq("k8v4 decode family", static_cast<int>(plan.family),
                  static_cast<int>(K8V4KvFamily::Grouped));
        expect_eq("k8v4 decode query_tile", plan.query_tile, 1);
        expect_eq("k8v4 decode target", plan.partition.target, 1);
        expect_eq("k8v4 decode capacity", plan.partition.capacity, 1);
        expect_eq("k8v4 decode key_shift", plan.partition.key_shift, 7);
    }
    // sms=2, heads=16, W=8 -> no underfill (filled 2 >= 1), single-CTA-per-SM budget.
    {
        const auto plan = make_k8v4_kv_causal_plan(16, 8, 1, envelop(1, 4096), 2);
        expect_eq("k8v4 grouped target", plan.partition.target, 1);
        expect_eq("k8v4 grouped capacity", plan.partition.capacity, 1);
        expect_eq("k8v4 grouped key_shift", plan.partition.key_shift, 7);
    }
    // sms=8, heads=24, W=9 -> ParallelGrouped, W<=16 halves the query tile.
    {
        const auto plan = make_k8v4_kv_causal_plan(24, 9, 1, envelop(1, 4096), 8);
        expect_eq("k8v4 parallel family", static_cast<int>(plan.family),
                  static_cast<int>(K8V4KvFamily::ParallelGrouped));
        expect_eq("k8v4 parallel query_tile", plan.query_tile, 5);
        expect_eq("k8v4 parallel target", plan.partition.target, 1);
        expect_eq("k8v4 parallel capacity", plan.partition.capacity, 1);
    }
    // sms=170, heads=24, W=48 -> 24 independent tiles, one CTA per SM, target 7.
    {
        const auto plan = make_k8v4_kv_causal_plan(24, 48, 1, envelop(1, 4096), 170);
        expect_eq("k8v4 wide query_tile", plan.query_tile, 8);
        expect_eq("k8v4 wide target", plan.partition.target, 7);
        expect_eq("k8v4 wide capacity", plan.partition.capacity, 7);
    }
    // Tiled branch: same width, the SM count drives the selected split count.
    {
        const auto small = make_k8v4_kv_causal_plan(16, 81, 1, envelop(1, 4096), 2);
        expect_eq("k8v4 tiled small family", static_cast<int>(small.family),
                  static_cast<int>(K8V4KvFamily::Tiled));
        expect_eq("k8v4 tiled small target", small.partition.target, 1);
        expect_eq("k8v4 tiled small capacity", small.partition.capacity, 1);
        const auto large = make_k8v4_kv_causal_plan(16, 81, 1, envelop(1, 4096), 170);
        expect_eq("k8v4 tiled large target", large.partition.target, 8);
        expect_eq("k8v4 tiled large capacity", large.partition.capacity, 8);
        expect_eq("k8v4 tiled key_shift", large.partition.key_shift, 9);
    }
}

// --- int8 -------------------------------------------------------------------

void check_int8() {
    // W<=4 forces the two-CTA-per-SM budget regardless of underfill.
    {
        const auto plan = make_int8_kv_causal_plan(16, 2, 1, envelop(1, 4096), 4);
        expect_eq("int8 narrow target", plan.partition.target, 4);
        expect_eq("int8 narrow capacity", plan.partition.capacity, 4);
    }
    {
        const auto plan = make_int8_kv_causal_plan(16, 8, 1, envelop(1, 4096), 4);
        expect_eq("int8 grouped family", static_cast<int>(plan.family),
                  static_cast<int>(Int8KvFamily::Grouped));
        expect_eq("int8 grouped target", plan.partition.target, 2);
        expect_eq("int8 grouped capacity", plan.partition.capacity, 2);
    }
    {
        const auto plan = make_int8_kv_causal_plan(24, 16, 2, envelop(1, 4096), 170);
        expect_eq("int8 parallel family", static_cast<int>(plan.family),
                  static_cast<int>(Int8KvFamily::ParallelGrouped));
        expect_eq("int8 parallel target", plan.partition.target, 21);
        expect_eq("int8 parallel capacity", plan.partition.capacity, 16);
    }
}

// --- fp8 --------------------------------------------------------------------

void check_fp8() {
    {
        const auto plan = make_fp8_kv_causal_plan(24, 8, 1, envelop(1, 4096), 170);
        expect_eq("fp8 grouped family", static_cast<int>(plan.family),
                  static_cast<int>(Fp8KvFamily::Grouped));
        expect_eq("fp8 grouped target", plan.partition.target, 42);
        expect_eq("fp8 grouped capacity", plan.partition.capacity, 16);
    }
    {
        const auto plan = make_fp8_kv_causal_plan(16, 81, 1, envelop(1, 4096), 170);
        expect_eq("fp8 tiled family", static_cast<int>(plan.family),
                  static_cast<int>(Fp8KvFamily::Tiled));
        expect_eq("fp8 tiled target", plan.partition.target, 8);
        expect_eq("fp8 tiled capacity", plan.partition.capacity, 8);
    }
}

// --- nvfp4 ------------------------------------------------------------------

void check_nvfp4() {
    {
        const auto plan = make_nvfp4_kv_causal_plan(24, 1, 1, envelop(1, 4096), 2);
        expect_eq("nvfp4 decode query_tile", plan.query_tile, 1);
        expect_eq("nvfp4 decode target", plan.partition.target, 1);
        expect_eq("nvfp4 decode capacity", plan.partition.capacity, 1);
        expect_eq("nvfp4 decode key_shift", plan.partition.key_shift, 7);
    }
    {
        const auto plan = make_nvfp4_kv_causal_plan(16, 16, 1, envelop(1, 4096), 8);
        expect_eq("nvfp4 parallel family", static_cast<int>(plan.family),
                  static_cast<int>(Nvfp4KvFamily::ParallelGrouped));
        expect_eq("nvfp4 parallel query_tile", plan.query_tile, 8);
        expect_eq("nvfp4 parallel target", plan.partition.target, 2);
        expect_eq("nvfp4 parallel capacity", plan.partition.capacity, 2);
    }
    // row_tiles=3 with 16 independent tiles underfills 8 SMs (filled 0 < 7), doubling the budget.
    {
        const auto plan = make_nvfp4_kv_causal_plan(24, 32, 1, envelop(1, 4096), 8);
        expect_eq("nvfp4 underfill target", plan.partition.target, 1);
        expect_eq("nvfp4 underfill capacity", plan.partition.capacity, 1);
    }
}

// --- bf16 -------------------------------------------------------------------

void check_bf16() {
    // Decode instance (16 query rows), normal target 1.
    {
        const auto plan = make_bf16_kv_causal_plan(24, 1, 1, envelop(1, 4096), 2);
        expect_eq("bf16 decode instance", static_cast<int>(plan.instance),
                  static_cast<int>(Bf16KvInstance::GroupedDecode));
        expect_eq("bf16 decode normal_target", plan.partition.normal_target, 1);
        expect_eq("bf16 decode long_target", plan.partition.long_target, 1);
        expect_eq("bf16 decode capacity", plan.partition.capacity, 1);
        expect_eq("bf16 decode key_rows", plan.partition.key_rows, 32);
    }
    // W=16, H=16 -> Grouped64; SM count scales the split budget.
    {
        const auto small = make_bf16_kv_causal_plan(16, 16, 1, envelop(1, 4096), 2);
        expect_eq("bf16 small instance", static_cast<int>(small.instance),
                  static_cast<int>(Bf16KvInstance::Grouped64));
        expect_eq("bf16 small normal_target", small.partition.normal_target, 1);
        expect_eq("bf16 small long_target", small.partition.long_target, 1);
        expect_eq("bf16 small capacity", small.partition.capacity, 1);
        const auto large = make_bf16_kv_causal_plan(16, 16, 1, envelop(1, 4096), 8);
        expect_eq("bf16 large normal_target", large.partition.normal_target, 4);
        expect_eq("bf16 large long_target", large.partition.long_target, 4);
        expect_eq("bf16 large capacity", large.partition.capacity, 4);
    }
    // W>256 selects the tiled instance and bypasses the SM-derived partition.
    {
        const auto plan = make_bf16_kv_causal_plan(16, 1024, 1, envelop(1, 16384), 170);
        expect_eq("bf16 tiled instance", static_cast<int>(plan.instance),
                  static_cast<int>(Bf16KvInstance::Tiled128));
        expect_bool("bf16 tiled grouped()", plan.grouped(), false);
    }
}

} // namespace

int main() {
    check_underfill_helper();
    check_partition_target_helper();
    check_k8v4();
    check_int8();
    check_fp8();
    check_nvfp4();
    check_bf16();

    std::cout << (g_failures == 0 ? "OK" : "FAIL") << " causal partition plan geometry\n";
    return g_failures == 0 ? 0 : 1;
}
