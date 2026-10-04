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
