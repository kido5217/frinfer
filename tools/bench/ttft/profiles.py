"""Executable Serve profiles for the external TTFT campaigns."""

from __future__ import annotations

from decimal import Decimal


# qwen3_8_27b_nvfp4: StateImageHostLayout sums 48 GDN conv/state regions and
# continuation hidden, each aligned to 256 B (state/state_image.cpp). DFlash2 also
# carries 5 local K/V layers of 2048 positions. MTP Full KV is in the KV arena.
STATE_IMAGE_BYTES = 153_954_304
DFLASH2_STATE_IMAGE_BYTES = 195_897_344


def _host_mib(capacity_bytes: int) -> str:
    return str(Decimal(capacity_bytes) / Decimal(1 << 20))


def _args(*values: str | int) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


COMMON_ARGS = _args(
    "--kv-dtype",
    "fp8",
    "--no-thinking",
    "--greedy",
    "--log-stats-interval-ms",
    1000,
)


_ALL_PROFILE_ARGS: dict[str, tuple[str, ...]] = {
    "text-cold-8k": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "text-cold-64k": _args(
        "--max-context", 65536,
        "--kv-capacity", 65536,
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "text-cold-256k": _args(
        "--max-context", 262144,
        "--kv-capacity", 262144,
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "cache-hot": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 1,
        "--device-state-slots", 2,
        "--host-context-mib", 0,
    ),
    "cache-state-working-set": _args(
        "--max-context", 32768,
        "--kv-capacity", 32768,
        "--max-concurrency", 1,
        "--prefill-chunk", 1024,
        "--spec", "dflash2",
        "--draft-tokens", 7,
        "--lm-head-draft",
        "--device-state-slots", 8,
        "--host-context-mib", _host_mib(8 * DFLASH2_STATE_IMAGE_BYTES),
    ),
    "cache-private-working-set": _args(
        "--max-context", 32768,
        "--kv-capacity", 32768,
        "--max-concurrency", 1,
        "--prefill-chunk", 1024,
        "--spec", "dflash2",
        "--draft-tokens", 7,
        "--lm-head-draft",
        "--device-state-slots", 2,
        "--host-context-mib", _host_mib(2 * DFLASH2_STATE_IMAGE_BYTES),
    ),
    "cache-pressure-device": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--device-state-slots", 2,
        "--host-context-mib", 0,
    ),
    "cache-pressure-state-host": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--device-state-slots", 0,
        "--host-context-mib", _host_mib(4 * STATE_IMAGE_BYTES),
    ),
    "cache-pressure-kv-host": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 2,
        "--device-state-slots", 2,
        "--host-context-mib", 8192,
    ),
    "cache-swap-64k-host": _args(
        "--max-context", 65536,
        "--kv-capacity", 65536,
        "--max-concurrency", 2,
        "--device-state-slots", 4,
        "--host-context-mib", 4608,
    ),
    "cache-rotation-55k-host": _args(
        "--max-context", 240000,
        "--kv-capacity", 240000,
        "--max-concurrency", 4,
        "--max-pending-requests", 32,
        "--pending-timeout-ms", 120000,
        "--spec", "mtp",
        "--draft-tokens", 3,
        "--lm-head-draft",
        "--device-state-slots", 2,
        "--host-context-mib", _host_mib((49152 << 20) + 24 * STATE_IMAGE_BYTES),
    ),
    "cache-pressure-both-host": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 2,
        "--device-state-slots", 0,
        "--host-context-mib", _host_mib((8192 << 20) + 4 * STATE_IMAGE_BYTES),
    ),
    "cache-pressure-evict": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 2,
        "--device-state-slots", 1,
        "--host-context-mib", 0,
    ),
    "cache-off": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "shared-prefix": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--device-state-slots", 2,
        "--host-context-mib", 0,
    ),
    "prefix-competition": _args(
        "--max-context", 16384,
        "--kv-capacity", 16384,
        "--max-concurrency", 1,
        "--device-state-slots", 3,
        "--host-context-mib", 0,
    ),
    "session-order": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--device-state-slots", 8,
        "--host-context-mib", 0,
    ),
    "scheduler-overlap": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--prefill-chunk", 1024,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "scheduler-prefill-128": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "scheduler-prefill-4096": _args(
        "--max-context", 8192,
        "--kv-capacity", 16384,
        "--max-concurrency", 2,
        "--prefill-chunk", 4096,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "scheduler-kv-pressure": _args(
        "--max-context", 7744,
        # Compliant full-resident pool: 2 lanes * ceil(7744/64)=121 pages = 242 pages.
        "--kv-capacity", 15488,
        "--max-concurrency", 2,
        "--pending-timeout-ms", 120000,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "lane-limit-8": _args(
        "--max-context", 4224,
        "--kv-capacity", 33792,
        "--max-concurrency", 8,
        "--max-pending-requests", 1,
        "--pending-timeout-ms", 120000,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "pending-timeout": _args(
        "--max-context", 4224,
        "--kv-capacity", 4224,
        "--max-concurrency", 1,
        "--max-pending-requests", 1,
        "--pending-timeout-ms", 100,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "context-boundary": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 1,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "vision-cache": _args(
        "--max-context", 32768,
        "--kv-capacity", 32768,
        "--max-concurrency", 1,
        "--device-state-slots", 2,
        "--vision",
        "--media-cache-mib", 512,
        "--media-live-mib", 512,
        "--host-context-mib", 0,
    ),
    "vision-thread-1": _args(
        "--max-context", 32768,
        "--kv-capacity", 32768,
        "--max-concurrency", 1,
        "--device-state-slots", 2,
        "--vision",
        "--media-cache-mib", 512,
        "--media-live-mib", 512,
        "--media-preprocess-threads", 1,
        "--host-context-mib", 0,
    ),
    "vision-concurrent": _args(
        "--max-context", 32768,
        "--kv-capacity", 65536,
        "--max-concurrency", 2,
        "--vision",
        "--media-cache-mib", 512,
        "--media-live-mib", 1024,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "media-cache-tight": _args(
        "--max-context", 8192,
        "--kv-capacity", 8192,
        "--max-concurrency", 1,
        "--vision",
        "--media-cache-mib", 16,
        "--media-live-mib", 128,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "vision-boundary": _args(
        "--max-context", 65536,
        "--kv-capacity", 65536,
        "--max-concurrency", 1,
        "--vision",
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "mixed-four": _args(
        "--max-context", 8192,
        "--kv-capacity", 32768,
        "--max-concurrency", 4,
        "--vision",
        "--host-context-mib", _host_mib((8192 << 20) + 8 * STATE_IMAGE_BYTES),
    ),
    "preemption-replay": _args(
        "--max-context", 512,
        "--kv-capacity", 512,
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--no-prefix-reuse",
        "--host-context-mib", 0,
    ),
    "preemption-snapshot": _args(
        "--max-context", 512,
        "--kv-capacity", 512,
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--no-prefix-reuse",
        "--host-context-mib", 512,
    ),
    "shared-growth-fairness": _args(
        "--max-context", 2688,
        "--kv-capacity", 2688,
        "--max-concurrency", 3,
        "--prefill-chunk", 128,
        "--device-state-slots", 2,
        "--host-context-mib", 0,
    ),
    "snapshot-history-cancel": _args(
        "--max-context", 512,
        "--kv-capacity", 512,
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--device-state-slots", 0,
        "--host-context-mib", 256,
    ),
    "vision-growth-replay": _args(
        "--max-context", 1024,
        "--kv-capacity", 1024,
        "--max-concurrency", 2,
        "--prefill-chunk", 128,
        "--device-state-slots", 0,
        "--host-context-mib", 0,
        "--no-prefix-reuse",
        "--vision",
        "--media-cache-mib", 0,
        "--media-live-mib", 64,
    ),
}

# The pinned no-preemption contract requires an explicit Main KV pool to equal
# max_concurrency * ceil(max_context / 64) -- the full-resident bound enforced by
# src/models/qwen3_5/program/planning/startup.cpp. Profiles that predate the bound and cannot be
# reframed under it are retired here with their reason; the campaign runner refuses them and
# validates every runnable profile against the bound before it starts a server.
KV_PAGE_TOKENS = 64

RETIRED_PROFILES: dict[str, str] = {
    "preemption-replay": "active-request preemption is pinned off",
    "preemption-snapshot": "active-request preemption is pinned off",
    "shared-growth-fairness": "active-request preemption is pinned off",
    "snapshot-history-cancel": "needs a paused resident (snapshot restore); preemption is pinned off",
    "vision-growth-replay": "needs Vision preemption/replay; preemption is pinned off",
    "cache-pressure-kv-host":
        "the frozen graph cannot reach Host-KV pressure under a compliant pool",
    "cache-pressure-evict":
        "the frozen graph cannot reach checkpoint-eviction pressure under a compliant pool",
    "cache-pressure-both-host":
        "the frozen graph cannot reach State+KV pressure under a compliant pool",
    "cache-swap-64k-host":
        "the frozen two-session swap cannot reach Host-KV pressure under a compliant pool",
    "cache-rotation-55k-host":
        "the frozen six-session rotation cannot reach Device/Host KV pressure under a compliant pool",
}

PROFILE_ARGS: dict[str, tuple[str, ...]] = {
    name: arguments for name, arguments in _ALL_PROFILE_ARGS.items()
    if name not in RETIRED_PROFILES
}


def page_count(capacity: int) -> int:
    """Physical Main KV pages for a token capacity, matching startup.cpp's ceil rule."""

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    return 1 + (capacity - 1) // KV_PAGE_TOKENS


def _option_value(arguments: tuple[str, ...], option: str) -> str | None:
    result = None
    for index, argument in enumerate(arguments):
        if argument == option and index + 1 < len(arguments):
            result = arguments[index + 1]
        elif argument.startswith(f"{option}="):
            result = argument.partition("=")[2]
    return result


def kv_capacity_error(common_args: tuple[str, ...], profile_args: tuple[str, ...]) -> str | None:
    """Diagnostic when an explicit --kv-capacity violates the full-resident bound.

    Returns None when the pool is compliant or when --kv-capacity is omitted; the automatic
    policy is resolved from the memory budget at startup and cannot be checked statically.
    """

    arguments = (*common_args, *profile_args)
    capacity = _option_value(arguments, "--kv-capacity")
    max_context = _option_value(arguments, "--max-context")
    if capacity is None or max_context is None:
        return None
    try:
        requested = page_count(int(capacity))
        context = int(max_context)
        concurrency = int(_option_value(arguments, "--max-concurrency") or "1")
    except (TypeError, ValueError):
        return None
    required = concurrency * page_count(context)
    if int(capacity) >= context and requested == required:
        return None
    return (
        f"kv_capacity={capacity} ({requested} pages) must equal the full-resident pool "
        f"{required} pages ({required * KV_PAGE_TOKENS} tokens) for {concurrency} lanes at "
        f"max_context={context}"
    )
