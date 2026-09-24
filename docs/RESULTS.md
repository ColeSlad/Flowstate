# Results

## Status

Phase 1 is complete: scalar, AVX2, and CUDA correctness checks pass, and the
measurements below demonstrate a CPU/GPU crossover. Explicit backend batches
are measured here; runtime microbatching, heuristic, and Jev evaluations remain
pending their implementation phases.

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

## CPU/GPU matrix — 2026-09-23

- Razer: Intel i9-12900H (14 physical cores / 20 logical processors), 32 GB host
  RAM, RTX 3080 Ti Laptop GPU with 16 GB VRAM, NVIDIA driver 596.21.
- Windows 11 Home 26200.9457; WSL 2.7.14; Ubuntu 24.04.5;
  Linux 6.18.33.2-microsoft-standard-WSL2. WSL exposes 20 logical CPUs.
- GCC 13.3.0, CMake 3.28.3, CUDA 13.2.86, target `sm_86` / `compute_86`.
- C++ flags: `-O3 -DNDEBUG -std=c++20 -ffp-contract=off`;
  CUDA: `-O3 -DNDEBUG -std=c++20 --fmad=false`. No fast-math.
- Seed 42, float32 dot product, K=10, one calling thread, 5 warmup batches,
  50 measured batches per case. CPU batches execute serially.
- Total batch wall time includes upload, kernel, download, and host top-K;
  dataset generation and the persistent dataset upload are excluded.
- Source revision: `6550d9bd6c841bef9e8f6157ef17cb35083ed9c1`, clean working tree.
  The code is unchanged from the initial compute implementation.

All measured cases are included below. Times are mean milliseconds for the entire
batch; CUDA QPS counts individual queries. Backend checksums matched in every case.

| Vectors | Dim | Batch | Scalar ms | AVX2 ms | CUDA ms | CUDA p95 ms | CUDA QPS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 384 | 1 | 0.307 | 0.028 | 0.123 | 0.159 | 8099.0 |
| 1,000 | 384 | 8 | 2.101 | 0.241 | 0.166 | 0.208 | 48072.0 |
| 1,000 | 384 | 32 | 8.089 | 0.953 | 0.482 | 0.515 | 66444.6 |
| 10,000 | 384 | 1 | 2.508 | 0.294 | 0.200 | 0.213 | 5007.6 |
| 10,000 | 384 | 8 | 20.835 | 2.850 | 0.931 | 0.961 | 8590.9 |
| 10,000 | 384 | 32 | 84.405 | 12.639 | 3.699 | 4.089 | 8650.5 |
| 100,000 | 128 | 1 | 7.255 | 2.902 | 0.997 | 1.050 | 1003.3 |
| 100,000 | 128 | 8 | 53.938 | 19.984 | 6.294 | 7.069 | 1271.0 |
| 100,000 | 128 | 32 | 219.991 | 82.500 | 23.237 | 28.099 | 1377.1 |
| 100,000 | 384 | 1 | 26.542 | 6.982 | 1.102 | 1.132 | 907.5 |
| 100,000 | 384 | 8 | 212.557 | 59.725 | 7.442 | 8.377 | 1074.9 |
| 100,000 | 384 | 32 | 856.508 | 240.664 | 26.526 | 27.807 | 1206.3 |
| 100,000 | 768 | 1 | 58.303 | 15.712 | 1.123 | 1.183 | 890.6 |
| 100,000 | 768 | 8 | 469.149 | 120.222 | 7.489 | 8.086 | 1068.2 |
| 100,000 | 768 | 32 | 1874.532 | 471.341 | 27.596 | 29.152 | 1159.6 |

At 1,000 vectors × 384, AVX2 is about 4.39× faster for a single query, while
CUDA is about 1.45× faster for batch 8 and 1.98× faster for batch 32. This is the
required crossover, using the same dataset and metric. At the default 100,000 ×
384 size, CUDA already wins single-query latency; increasing batch size primarily
amortizes transfer and launch overhead. These results do not imply that every
workload benefits from waiting to form a batch.

Representative CUDA event intervals (dimension 384):

| Vectors | Batch | Upload ms | Kernel ms | Download ms | Total wall ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1 | 0.028 | 0.013 | 0.052 | 0.123 |
| 1,000 | 32 | 0.034 | 0.237 | 0.098 | 0.482 |
| 10,000 | 1 | 0.028 | 0.081 | 0.056 | 0.200 |
| 10,000 | 32 | 0.082 | 2.366 | 0.365 | 3.699 |
| 100,000 | 1 | 0.030 | 0.729 | 0.175 | 1.102 |
| 100,000 | 32 | 0.057 | 19.710 | 1.475 | 26.526 |

Wall time also includes CPU selection, buffer preparation, and synchronization;
it need not equal the sum of the device event intervals. Host buffers are pageable.

Reproduction on the configured GPU host:

```sh
export PATH=/usr/local/cuda-13.2/bin:/usr/lib/wsl/lib:$PATH
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DFLOWSTATE_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-gpu -j8
ctest --test-dir build-gpu --output-on-failure
python3 benchmarks/run_matrix.py --binary build-gpu/flowstate_bench --require-all \
  --iterations 50 --warmup 5 --output benchmark-output/razer-100k
# Repeat for --vectors 1000 and 10000 with --dimensions 384 and separate outputs.
```

The raw CSV and compiler/hardware metadata are retained in ignored
`benchmark-output/razer-*` directories on both development and validation hosts.
An SSH interruption occurred after the complete 100k sweep and two small-workload
cells; completed rows were preserved and only the missing cells were run afterward.
The resumed 1k output is marked in its metadata. This was one serial matrix run,
without CPU affinity, locked clocks, or a controlled Windows desktop workload.
Small-sample p95/p99 statistics and WSL results should not be generalized to
other GPUs, sustained production traffic, or a bare-metal Linux system.

## GPU-host validation

- Release CUDA build succeeds with all five CTest tests passing, no skips.
  Explicit AVX2 and CUDA suites pass 44,845 and 45,133 checks respectively.
- Linux AddressSanitizer and UBSan CPU builds pass, including AVX2; the CUDA suite
  is intentionally unavailable in these CPU-only sanitizer builds.
- Compute Sanitizer 2026.1.1 cannot initialize the WDDM debugger interface. It
  reports device unsupported and asks for `EnableDebuggerInterface.bat`; the
  CUDA test itself passes, but this is **not** a passing GPU memory-sanitizer run.
  No machine-wide GPU debugger settings were changed.
- The ordinary CUDA suite covers exact top-K, tail dimensions, invalid inputs,
  independent reference equivalence, batch buffer growth/reuse, and error recovery.
- Phase 1 review found no outstanding correctness issues. The first hardware
  validation required no changes to the backend implementation.

Phase 2 must now verify concurrent submission, bounded queues, safe shutdown,
size/timeout microbatch flushes, and a runtime batching throughput improvement.
