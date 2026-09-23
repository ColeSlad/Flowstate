# Results

## Status

Phase 1 remains incomplete. There is no measured AVX2/GPU comparison, batching
speedup, or CPU/GPU crossover yet. Runtime, heuristic, and Jev evaluations are
pending their implementation phases. No speedup is claimed from the data below.

## Scalar baseline — 2026-09-23

- Apple M3 Max, 16 CPU cores (12 performance + 4 efficiency), 64 GB RAM.
- macOS 26.6.2; native ARM64 execution; Apple Clang 17.0.0
  (`clang-1700.3.19.1`); CMake 4.3.3.
- Release compute flags: `-O3 -DNDEBUG -std=c++20 -ffp-contract=off`.
  No fast-math. Warnings: `-Wall -Wextra -Wpedantic`.
- NVIDIA GPU and CUDA toolkit: unavailable; the integrated Apple GPU is not CUDA.
- Dataset: 100,000 vectors, float32, seed 42, top-K 10, dot product.
- Batch size 1, one calling thread, 5 warmups, 50 measured queries per dimension.
- Deterministic query reused across iterations; no CPU affinity or clock locking.
- Dataset generation is excluded; dot products and exact host top-K are included.

| Dimension | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Queries/sec |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 4.660 | 4.648 | 4.959 | 5.155 | 214.58 |
| 384 | 18.807 | 18.750 | 19.388 | 19.630 | 53.17 |
| 768 | 47.946 | 47.739 | 50.622 | 54.178 | 20.86 |

Reproduction:

```sh
python3 benchmarks/run_matrix.py --batch-sizes 1 --iterations 50 --warmup 5 \
  --label 'Apple M3 Max; native arm64; 16 CPU cores; 64 GB RAM' \
  --output benchmark-output/m3max-arm64
```

Raw CSV and compiler metadata are retained locally under the ignored output
path. The sandbox prevented `sysctl` CPU identification in the runner; the model
and memory were verified with `system_profiler` and included in the run label.
The benchmark was collected from the initial Phase 1 working tree before its
first commit. Short runs characterize a local baseline, not sustained load or
production tail latency. Fifty samples are insufficient for a strong p99 claim;
p99 here is the nearest-rank sample statistic.

## Validation and review

- Native ARM64 Release: scalar suite and two CLI tests pass; AVX2/CUDA suites
  explicitly skip. The scalar suite executes 44,857 checks.
- x86-64 Release cross-build: builds and runs under Rosetta; scalar and CLI tests
  pass. AVX2 remains unavailable according to CPU/OS feature detection. No Rosetta
  timings are presented as native x86 performance.
- Native Debug UBSan: scalar and CLI tests pass without diagnostics; AVX2/CUDA
  suites skip.
- Native Debug ASan: build succeeds, but the test hangs during sanitizer startup
  before `main()`. A process sample shows recursive `AsanInitInternal` through
  `get_dyld_hdr` / `dyld_shared_cache_iterate_text_swift` / allocator initialization,
  spinning in `StaticSpinMutex::LockSlow`. A retry outside the sandbox also
  timed out after 15 seconds. This is not a passing ASan check.
- Explicit CUDA configuration fails with the expected missing-`nvcc` diagnostic.
- Manual CLI failure checks pass for unsupported backends, mandatory backend
  availability, negative/missing numbers, invalid top-K, seed overflow, unknown
  backend, and zero warmup.
- Implementation review checked heap ordering, SIMD tail bounds and dispatch,
  ownership, CUDA synchronization/error paths, and benchmark timing boundaries.
  Added per-test timeouts after the ASan startup stall. No unresolved CPU
  correctness findings; CUDA review cannot substitute for compilation/execution.

## Required next validation

On an x86-64 AVX2 host with an NVIDIA GPU and a C++20 CUDA toolkit:

```sh
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DFLOWSTATE_ENABLE_CUDA=ON
cmake --build build-gpu -j
ctest --test-dir build-gpu --output-on-failure
./build-gpu/flowstate_tests avx2
./build-gpu/flowstate_tests cuda
python3 benchmarks/run_matrix.py --binary build-gpu/flowstate_bench \
  --require-all --label 'record CPU and GPU models' --output benchmark-output/gpu
```

Fix any CUDA compilation/runtime findings, verify equivalence without skipped
backends, and record scalar/AVX2/CUDA results at dimensions 128/384/768 and batch
sizes 1/8/32. Inspect upload, kernel, download, and full-result times. If no
crossover occurs, profile the measured bottleneck before changing the algorithm;
do not alter the workload merely to manufacture a win. Repeat ASan with a working
sanitizer toolchain. Only then complete Phase 1 and proceed to the runtime.
