# FrInfer Jinja source base

This maintained implementation originates from [llama.cpp](https://github.com/ggml-org/llama.cpp),
commit [`7609846557c50f9d984719a9e1e8c5f3d02f807b`](https://github.com/ggml-org/llama.cpp/commit/7609846557c50f9d984719a9e1e8c5f3d02f807b),
`common/jinja/`, under the MIT license; see [LICENSE](LICENSE).

FrInfer maintains this fork for chat-template rendering and adopts upstream fixes selectively.

Unicode case data is generated from Unicode 17.0.0 `SpecialCasing.txt` and
`DerivedCoreProperties.txt`, under [UNICODE-LICENSE](UNICODE-LICENSE):

```bash
python3 tools/generate_jinja_unicode.py /path/to/unicode-17.0.0
```

## Adopted upstream fixes

Selected post-base upstream fixes adopted into the fork, applied by hand to the
FrInfer-reformatted sources and verified by the fork's test suite plus the
`NINFER_PARITY_TEMPLATE` template-parity gate.

| Upstream | File | Change |
|---|---|---|
| [`def4d406a`](https://github.com/ggml-org/llama.cpp/commit/def4d406ae2c2f39573120d68730fbb7760b24bf) (`#29776`) | `jinja/runtime.cpp` | `for_statement::execute_impl`: construct the per-iteration `context loop_scope(scope)` copy only when the loop carries a `select`/`test` filter, instead of on every iteration. Semantics-preserving; unfiltered loops skip one context copy per iteration. |
