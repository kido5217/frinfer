#!/usr/bin/env python3
"""Independent FP64 oracle for the logprob_topk prototype (THROWAWAY).

Reads the raw little-endian files the CUDA driver wrote, decodes BF16 by hand,
recomputes the §2.2 formula and the top-20 selection in NumPy float64, and
compares against the kernel outputs. Never calls the CUDA code.

Usage: python3.13 oracle.py <data-dir>
Prints human-readable detail plus a final machine-parsable ORACLE line.
"""
import os
import sys

import numpy as np

K = 20


def read_meta(path):
    meta = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            meta[key] = parts[1:]
    return meta


def main():
    if len(sys.argv) != 2:
        print("usage: oracle.py <data-dir>", file=sys.stderr)
        return 2
    d = sys.argv[1]
    meta = read_meta(os.path.join(d, "meta.txt"))
    case = meta["case"][0]
    V = int(meta["V"][0])
    rows = int(meta["rows"][0])
    batch_rows = int(meta["batch_rows"][0])
    width = int(meta["width"][0])
    temp = np.array([float(x) for x in meta["temperature"]], dtype=np.float64)
    pres = np.array([float(x) for x in meta["presence"]], dtype=np.float64)
    freq = np.array([float(x) for x in meta["frequency"]], dtype=np.float64)
    valid = np.array([int(x) for x in meta["valid_per_row"]], dtype=np.int64)

    # --- decode BF16 by hand: u16 -> uint32 << 16 -> float32 ---
    raw = np.fromfile(os.path.join(d, "logits.bf16"), dtype="<u2")
    assert raw.size == V * rows, f"logits size {raw.size} != {V*rows}"
    u32 = (raw.astype(np.uint32) << 16)
    logits = u32.view(np.float32).reshape(V, rows).astype(np.float64)

    counts = np.fromfile(os.path.join(d, "counts.i32"), dtype="<i4")
    assert counts.size == rows * V
    counts = counts.reshape(rows, V).astype(np.float64)  # [rows, V]

    k_topv = np.fromfile(os.path.join(d, "top_values.f32"), dtype="<f4").reshape(K, rows).T.astype(np.float64)
    k_topi = np.fromfile(os.path.join(d, "top_ids.i32"), dtype="<i4").reshape(K, rows).T
    k_lse = np.fromfile(os.path.join(d, "lse.f32"), dtype="<f4").astype(np.float64)
    k_chosen = np.fromfile(os.path.join(d, "chosen.i32"), dtype="<i4")

    # --- recompute the formula in float64 ---
    T = np.where(temp > 0.0, temp, 1.0)          # [rows]
    counts_t = counts.T                            # [V, rows]
    adj = logits - pres[None, :] * (counts_t > 0) - freq[None, :] * counts_t
    scaled = adj / T[None, :]                      # [V, rows]

    m = scaled.max(axis=0)                         # [rows]
    if not np.all(np.isfinite(m)):
        print(f"ORACLE dir={d} case={case} INVALID all-masked row", file=sys.stderr)
        return 1
    sumexp = np.exp(scaled - m[None, :]).sum(axis=0)
    lse = m + np.log(sumexp)
    logprob = scaled - lse[None, :]

    # top-20: scaled descending; stable argsort on -scaled keeps lower id on ties.
    order = np.argsort(-scaled, axis=0, kind="stable")   # [V, rows]
    topi = order[:K, :].T                                 # [rows, K]
    topv = np.take_along_axis(logprob, order[:K, :], axis=0).T

    # --- compare ---
    top20_exact = bool(np.array_equal(topi, k_topi))
    first = ""
    if not top20_exact:
        mism = np.argwhere(topi != k_topi)
        if mism.size:
            r, k = int(mism[0][0]), int(mism[0][1])
            first = (f"row={r} pos={k} oracle_id={int(topi[r,k])} kernel_id={int(k_topi[r,k])} "
                     f"oracle_v={topv[r,k]:.9g} kernel_v={k_topv[r,k]:.9g}")
        else:
            first = "shape-only"

    # max-abs and max-rel error over the K entries. When the id sequences match
    # exactly the kernel values are already in the oracle's order; otherwise
    # reorder the kernel by where its ids sit in the oracle order.
    if top20_exact:
        aligned_kernel = k_topv
    else:
        # Not the same id sequence: compare the descending value multisets so the
        # value error is still informative (top20_exact=0 already flags the miss).
        aligned_kernel = np.sort(k_topv, axis=1)[:, ::-1]
        topv = np.sort(topv, axis=1)[:, ::-1]
    diff = aligned_kernel - topv
    maxabs = float(np.max(np.abs(diff)))
    denom = np.maximum(np.abs(topv), 1e-30)
    maxrel = float(np.max(np.abs(diff) / denom))

    lse_maxabs = float(np.max(np.abs(k_lse - lse)))

    # chosen-token check: prototype sets chosen = top-1 id (no sampling).
    chosen_ok = bool(np.array_equal(k_chosen, topi[:, 0]))
    chosen_val_ok = bool(np.allclose(k_topv[:, 0], topv[:, 0], rtol=1e-5, atol=1e-5))
    chosen_ok = chosen_ok and chosen_val_ok

    print(f"case={case} batch_rows={batch_rows} width={width} rows={rows} V={V}")
    print(f"  top20 exact: {top20_exact}")
    if first:
        print(f"  first mismatch: {first}")
    print(f"  top_values max-abs err: {maxabs:.3e}")
    print(f"  top_values max-rel err: {maxrel:.3e}")
    print(f"  lse max-abs err:        {lse_maxabs:.3e}")
    print(f"  chosen-token check:     {chosen_ok}")
    print(f"ORACLE dir={d} case={case} batch_rows={batch_rows} width={width} rows={rows} "
          f"top20_exact={int(top20_exact)} maxabs={maxabs:.6e} maxrel={maxrel:.6e} "
          f"lse_maxabs={lse_maxabs:.6e} chosen_ok={int(chosen_ok)}")
    # Assertion: the top-20 id sequence and the chosen-token entry must match.
    return 0 if (top20_exact and chosen_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
