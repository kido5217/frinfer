#!/usr/bin/env python3
"""Aggregate the run matrix into results/summary.md (THROWAWAY)."""
import glob
import os
import re
import sys

DECODE_STEP_US = 13700.0  # survey baseline: ~13.7 ms per decode step


def parse_kv(line):
    out = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    oracle = {}
    bench = {}
    for p in glob.glob(os.path.join(root, "data", "*", "oracle.txt")):
        with open(p) as f:
            for line in f:
                if line.startswith("ORACLE "):
                    kv = parse_kv(line)
                    key = (kv["case"], int(kv["batch_rows"]), int(kv["width"]))
                    oracle[key] = kv
    for p in glob.glob(os.path.join(root, "data", "*", "bench.txt")):
        with open(p) as f:
            for line in f:
                if line.startswith("BENCH "):
                    kv = parse_kv(line)
                    key = (kv["case"], int(kv["batch_rows"]), int(kv["width"]))
                    bench[key] = kv

    order = ["plain-t1", "plain-t07", "greedy", "penalty", "masked", "mask-penalty", "spec-k4"]
    keys = sorted(oracle.keys(), key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1]))

    out = []
    out.append("# logprob_topk prototype — measured results\n")
    out.append(f"- Baseline decode step: **{DECODE_STEP_US/1000.0:.1f} ms** ({DECODE_STEP_US:.0f} µs), per the survey cited in the design doc §2.5.")
    out.append("- V = 151936 (Qwen3.5 vocab); each row is split across 128 CTAs (256 threads) in a partials kernel, then a one-CTA-per-row combine kernel merges the partials into the final top-20.")
    out.append("- Errors are kernel FP32 vs the independent NumPy FP64 oracle over BF16-decoded inputs.")
    out.append("")
    out.append("| case | batch_rows | width | eff_rows | top-20 exact | max-abs err | max-rel err | lse max-abs | chosen ok | mean µs/call | % of 13.7 ms |")
    out.append("|---|---:|---:|---:|:---:|---:|---:|---:|:---:|---:|---:|")

    all_ok = True
    for k in keys:
        o = oracle.get(k)
        if o is None:
            continue
        b = bench.get(k, {})
        exact = int(o["top20_exact"])
        chosen = int(o["chosen_ok"])
        if exact != 1 or chosen != 1:
            all_ok = False
        us = float(b["mean_us"]) if "mean_us" in b else float("nan")
        pct = 100.0 * us / DECODE_STEP_US
        out.append(
            f"| {k[0]} | {k[1]} | {k[2]} | {o['rows']} | {'yes' if exact else '**NO**'} | "
            f"{float(o['maxabs']):.3e} | {float(o['maxrel']):.3e} | {float(o['lse_maxabs']):.3e} | "
            f"{'yes' if chosen else '**NO**'} | {us:.2f} | {pct:.4f}% |"
        )
    out.append("")
    out.append(f"**Overall top-20 exact + chosen-token pass: {'PASS' if all_ok else 'FAIL'}**")
    out.append("")
    out.append("Notes:")
    out.append("- `spec-k4` is rows=8 with a k+1=5 column block modelled as a single 40-row-instance launch (the round's whole verify block).")
    out.append("- `mean µs/call` is device time for one Op launch, averaged over 100 iterations after 10 warm-up launches (CUDA events bracketing the loop).")
    out.append("- `% of 13.7 ms` is the implied fraction of one decode step for that case's row count.")
    out.append("- Design-sketch caveat: the design §2.3 sketch (one CTA per row with a block reduction) was implemented first and its best variant measured ~336 us for rows=1. An unsplit one-CTA-per-row layout exposes memory latency with only 8 warps per SM, so the split-CTA implementation above is ~7x faster and is the number reported in the table.")
    print("\n".join(out))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
