# FrInfer task runner.
#
# Run from inside the Nix devShell (`nix develop`), which provides just, cmake,
# ninja, nvcc and the test Python. These recipes wrap the commands documented in
# tests/README.md and bench/README.md; see docs/maintainer/build-system.md.

set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

# Default: run the test suite.
default: test

# Configure with the dev preset (products, tests, benchmarks) and build everything.
build:
    cmake --preset dev
    cmake --build build -j

# Run the CTest suite. Extra arguments forward to ctest, e.g. `just test -j8 -R sampling`.
test *args:
    ctest --preset dev --output-on-failure {{args}}

# Build one test target and re-run it through CTest: `just test-one ninfer_sampling_test`.
test-one name *args:
    cmake --build build --parallel --target {{name}}
    ctest --preset dev -R {{name}} --output-on-failure {{args}}

# Build and run a benchmark executable: `just bench ninfer_rmsnorm_bench --tokens 128`.
bench target *args:
    cmake --build build --parallel --target {{target}}
    build/bench/{{target}} {{args}}

# Full verification as one bounded command: configure + build, the whole CTest
# suite, then the hermetic Python suites (artifact, conversion, Serve TTFT, text,
# tool-schema and serve-corpus) with the independent jsonschema / re.fullmatch
# grammar oracles and the Qwen tool-schema constraint probe. Everything is tee-ed to
# build/verify.log and the run ends with an unambiguous `VERIFY OK` or
# `VERIFY FAIL (step: <step>)` line. Run inside the Nix devShell (see header).
verify:
    @mkdir -p build; \
    { \
      fail=; \
      echo '== configure =='; cmake --preset dev || fail=configure; \
      if [ -z "$fail" ]; then echo '== build =='; cmake --build build -j || fail=build; fi; \
      if [ -z "$fail" ]; then echo '== ctest =='; ctest --preset dev --output-on-failure || fail=ctest; fi; \
      if [ -z "$fail" ]; then echo '== pytest =='; python3.13 -m pytest tests/artifact tests/convert tests/bench/ttft tests/text tests/models/qwen3_5/test_tool_schema.py tests/test_serve_corpus.py -q || fail=pytest; fi; \
      if [ -z "$fail" ]; then echo 'VERIFY OK'; else echo "VERIFY FAIL (step: $fail)"; exit 1; fi; \
    } 2>&1 | tee build/verify.log
