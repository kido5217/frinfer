# logprob_topk prototype — measured results

- Baseline decode step: **13.7 ms** (13700 µs), per the survey cited in the design doc §2.5.
- V = 151936 (Qwen3.5 vocab); each row is split across 128 CTAs (256 threads) in a partials kernel, then a one-CTA-per-row combine kernel merges the partials into the final top-20.
- Errors are kernel FP32 vs the independent NumPy FP64 oracle over BF16-decoded inputs.

| case | batch_rows | width | eff_rows | top-20 exact | max-abs err | max-rel err | lse max-abs | chosen ok | mean µs/call | % of 13.7 ms |
|---|---:|---:|---:|:---:|---:|---:|---:|:---:|---:|---:|
| plain-t1 | 1 | 1 | 1 | yes | 1.523e-07 | 4.206e-08 | 1.523e-07 | yes | 45.73 | 0.3338% |
| plain-t1 | 8 | 1 | 8 | yes | 6.365e-07 | 1.575e-07 | 6.365e-07 | yes | 80.43 | 0.5871% |
| plain-t07 | 1 | 1 | 1 | yes | 1.059e-06 | 5.499e-07 | 2.493e-07 | yes | 48.00 | 0.3504% |
| plain-t07 | 8 | 1 | 8 | yes | 1.617e-06 | 1.311e-06 | 8.129e-07 | yes | 80.43 | 0.5871% |
| greedy | 1 | 1 | 1 | yes | 1.523e-07 | 4.206e-08 | 1.523e-07 | yes | 49.87 | 0.3640% |
| greedy | 8 | 1 | 8 | yes | 6.365e-07 | 1.575e-07 | 6.365e-07 | yes | 80.57 | 0.5881% |
| penalty | 1 | 1 | 1 | yes | 6.096e-07 | 1.215e-07 | 2.521e-07 | yes | 45.83 | 0.3345% |
| penalty | 8 | 1 | 8 | yes | 8.843e-07 | 1.830e-07 | 4.668e-07 | yes | 80.35 | 0.5865% |
| masked | 1 | 1 | 1 | yes | 3.981e-07 | 1.079e-07 | 3.981e-07 | yes | 25.36 | 0.1851% |
| masked | 8 | 1 | 8 | yes | 3.742e-07 | 1.549e-07 | 3.742e-07 | yes | 45.92 | 0.3352% |
| mask-penalty | 1 | 1 | 1 | yes | 2.584e-07 | 5.220e-08 | 1.970e-08 | yes | 23.57 | 0.1721% |
| mask-penalty | 8 | 1 | 8 | yes | 9.713e-07 | 2.516e-07 | 5.538e-07 | yes | 45.85 | 0.3347% |
| spec-k4 | 8 | 5 | 40 | yes | 6.993e-07 | 1.711e-07 | 6.993e-07 | yes | 228.49 | 1.6678% |

**Overall top-20 exact + chosen-token pass: PASS**

Notes:
- `spec-k4` is rows=8 with a k+1=5 column block modelled as a single 40-row-instance launch (the round's whole verify block).
- `mean µs/call` is device time for one Op launch, averaged over 100 iterations after 10 warm-up launches (CUDA events bracketing the loop).
- `% of 13.7 ms` is the implied fraction of one decode step for that case's row count.
- Design-sketch caveat: the design §2.3 sketch (one CTA per row with a block reduction) was implemented first and its best variant measured ~336 us for rows=1. An unsplit one-CTA-per-row layout exposes memory latency with only 8 warps per SM, so the split-CTA implementation above is ~7x faster and is the number reported in the table.
