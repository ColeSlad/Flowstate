# Results

## Status

Phases 1–4 are complete: scalar, AVX2, and CUDA correctness checks pass,
the measurements below demonstrate a CPU/GPU crossover, and concurrent runtime
batching improves burst throughput. The heuristic responds to workload changes. Local Laya integration and evaluation
are complete; its limitations and the static GPU baseline advantage are reported below.

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

## Concurrent runtime batching — 2026-09-24

Same Razer hardware, toolchain, Release flags, dataset seed 42, dimension 384,
and top-K 10 as the matrix above. Two CPU workers use AVX2; one GPU worker owns
CUDA. The waiting queue capacity is 1,024; GPU batches allow 32 requests with
a 2 ms oldest-request timeout. Each mode runs three warmup bursts followed by
ten measured bursts of 512 requests, cycling through 32 deterministic queries.
Each burst drains before the next starts. Setup and shutdown are excluded;
throughput includes submission and result collection. Request latency includes
queueing, execution, and publication up to immediately before promise fulfillment.
Source is the Phase 2 working tree based on `0a7546b`, after the timing review fix.

| Vectors | Route | Queries/sec | p50 ms | p95 ms | p99 ms | Mean queue wait ms |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | AVX2, 2 workers | 51,254.9 | 4.873 | 9.230 | 9.706 | 4.880 |
| 1,000 | GPU immediate | 6,752.3 | 38.135 | 72.157 | 76.111 | 37.827 |
| 1,000 | GPU batch | 57,331.9 | 4.593 | 8.522 | 8.957 | 4.109 |
| 100,000 | AVX2, 2 workers | 262.3 | 979.679 | 1,856.772 | 1,936.052 | 972.396 |
| 100,000 | GPU immediate | 1,047.9 | 244.613 | 464.529 | 484.013 | 243.735 |
| 100,000 | GPU batch | 1,272.1 | 203.732 | 399.316 | 404.935 | 188.469 |

Batching improves throughput over immediate GPU execution by **8.49×** on the
small dataset and **1.21×** on the default dataset. All checksums match within
each dataset. Both batch cases execute 160 full batches of 32 via size flushes;
the immediate cases execute 5,120 batches of one. Mean CUDA kernel intervals are
0.024/0.237 ms (immediate/batch) for 1k vectors and 0.571/18.853 ms for 100k.
These are whole-batch intervals, counted once per batch. Timeout flushing is
covered by tests; these saturated bursts produce no timeout flushes.

This is an intentionally queued burst workload, not a single-query latency
comparison. Tail latency includes hundreds of preceding requests, especially
on the large dataset. It does not establish an SLO for sustained open-loop
traffic, or a benefit from batching sparse arrivals. The later load generator
will measure those policy tradeoffs. Windows background load, WSL scheduling,
and unlocked clocks remain uncontrolled.

```sh
for vectors in 1000 100000; do
  python3 benchmarks/run_matrix.py --binary build-gpu/flowstate_runtime_bench \
    --require-all --vectors "$vectors" --dimensions 384 --batch-sizes 32 \
    --iterations 10 --warmup 3 --label 'Razer; WSL2; two CPU workers' \
    --output "benchmark-output/runtime-$vectors"
done
```

Raw CSV and hardware/compiler metadata are retained in the ignored output
directories on both hosts. The benchmark CLI also exposes worker count, queue
capacity, requests per burst, and batch timeout for direct runs.

Phase 2 validation and review:

- All seven Release CTest tests pass on the Razer, including mixed CPU/CUDA
  routing, batch size/timeout, concurrent submissions, shared queue capacity,
  concurrent shutdown calls, shutdown during submission and active GPU work,
  future lifetime, and backend failure recovery.
- Linux CPU-only AddressSanitizer, UBSan, and ThreadSanitizer suites pass without
  diagnostics; CUDA tests explicitly skip in sanitizer builds. GCC TSan initially
  failed before `main()` with an unexpected memory mapping under WSL. Running
  `setarch x86_64 -R ctest --test-dir build-thread --output-on-failure` resolves
  its address-space conflict for that process; no system-wide ASLR setting changed.
- Native ARM Release tests pass; unavailable AVX2/CUDA tests skip.
- The invalid-CLI test now checks both the expected exit code and diagnostic,
  so a sanitizer startup failure cannot masquerade as a passing negative test.
- The specified deeper `codex review --uncommitted` found that end-to-end latency
  stopped at backend completion. The timestamp now includes completion work and
  earlier batch publications, with regression coverage. The benchmark above was
  rerun after this fix. No unresolved review findings remain.


## Adaptive workload trace — 2026-09-24

Same Razer hardware and Release toolchain as above, now with NVML 13.2 headers
and the WSL NVIDIA driver library. Source is the Phase 3 working tree based on
`ce3ab45`, with real NVML sampling enabled. Every mode uses 100k × 384, seed 42,
two AVX2 workers, queue capacity 1,024, batch maximum 32, and a 2 ms timeout.
Three warmup batches run on each backend instance before runtime creation.

The unchanged `benchmarks/traces/adaptive.csv` schedules 8,040 queries over 20 s:
4 s at 10 QPS (K=10), 6 s at 1,200 QPS in bursts of 32 (K=10), 5 s at 150 QPS
(K=50), then 5 s at 10 QPS (K=10). Each mode receives the same generated queries
and intended arrival times. Measurements include the final drain. Summary
percentiles use exact nearest-rank samples, not the rolling histogram estimates.
SLO violations count offered latency over 15 ms plus every rejected/failed query.

| Mode | Completed / offered | Rejected | QPS | p50 ms | p95 ms | p99 ms | SLO violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static AVX2 | 3,335 / 8,040 | 4,705 | 166.7 | 3,484.386 | 4,129.681 | 4,143.897 | 7,972 |
| Static GPU immediate | 8,040 / 8,040 | 0 | 402.0 | 401.295 | 732.027 | 750.977 | 7,319 |
| Static GPU batch | 8,040 / 8,040 | 0 | 402.0 | 24.412 | 39.561 | 54.262 | 7,203 |
| Heuristic | 8,040 / 8,040 | 0 | 402.0 | 25.248 | 785.455 | 2,011.002 | 7,200 |

No backend failures or controller errors occurred. All modes that completed the
full trace have matching result checksums. The CPU checksum differs because
4,705 requests were rejected. Observed p99 including producer lateness was
4,144.032 / 751.051 / 54.331 / 2,011.106 ms in table order. Maximum producer
lateness was 11.58 / 5.81 / 21.71 / 0.97 ms; it is not hidden in runtime latency.

The heuristic made four transitions, visible in the 500 ms telemetry stream:

| Observed elapsed | Phase | Selected policy | 1 s arrival rate | Queue depth |
| ---: | --- | --- | ---: | ---: |
| 0 s | Low | CPU latency | 0 | 0 |
| 4.50 s | Burst | GPU batch | 612 | 500 |
| 11.00 s | Moderate | Balanced | 145 | 0 |
| 15.50 s | Low return | GPU immediate | 76 | 0 |
| 16.00 s | Low return | CPU latency | 9 | 0 |

It routed 991 queries to CPU and 7,049 to GPU. The first 500 ms of burst traffic
queued on CPU before the controller switched. Those accepted requests retain
their routes and explain the long tail; later GPU requests complete much sooner.
Static GPU batching is clearly better for tail latency on this large dataset.
Throughput ties among modes that serve all traffic because the trace's average
offered rate is 402 QPS. This run demonstrates adaptation, not a throughput win
for the heuristic or a satisfied 15 ms SLO. No thresholds were tuned to reverse
that result. Queue migration, hysteresis, and learned predictors remain out of
scope for the current phases.

Sampled system utilization (mean of available readings, including idle periods):

| Mode | CPU mean | GPU mean | GPU utilization samples | Maximum observed queue |
| --- | ---: | ---: | ---: | ---: |
| Static AVX2 | 6.75% | 0.57% | 21/41 | 1,023 |
| Static GPU immediate | 2.06% | 26.34% | 41/41 | 824 |
| Static GPU batch | 2.05% | 28.56% | 41/41 | 1 |
| Heuristic | 4.32% | 24.12% | 41/41 | 500 |

CPU is whole-system utilization over the 20 WSL logical CPUs, not the fraction
of two workers occupied. The heuristic has 40 CPU samples because its first
reading has no prior counter delta. NVML intermittently returned no GPU
utilization reading during the CPU run; these entries stay empty and are excluded
from the mean. Free GPU memory was available in all sampled rows (about 15.81 GB).
There was no injected GPU contention. Actual contention testing belongs to the
later demo/evaluation work; a pure scheduler test covers the high-utilization
branch without presenting test inputs as measured telemetry.

```sh
python3 benchmarks/run_load.py --binary build-gpu/flowstate_load \
  --cuda on --cpu-backend avx2 --output benchmark-output/phase3-100k \
  --label 'Razer; WSL2; NVML enabled'
```

The runner stores all four summaries, sampled telemetry, exact trace/hash,
commands, compiler flags, hardware, and Git state in the ignored output directory
on both hosts. One serial run per mode, in the table's order, is reported. This
is a short controlled trace with uncontrolled desktop load and clock speeds;
it is not a confidence interval or a universal comparison of schedulers.
An earlier development run lacked NVML detection and was retained separately;
the table reports the complete rerun after the detection fix.

Phase 3 validation:

- All eight CUDA Release CTest tests pass without skips. CPU ASan, UBSan, and
  TSan suites pass; TSan still uses the documented per-process ASLR workaround.
- New tests exercise histogram bounds, window expiry/reuse, missing readings,
  policy parsing, invalid thresholds, all heuristic branches, policy switching
  with queued work, concurrent telemetry reads, and trace validation.
- Native ARM Release tests pass with AVX2/CUDA explicitly skipped. A short CPU
  trace completes all 606 requests under static and heuristic modes with matching
  checksums and no rejections/errors.
- Manual phase review checked bounded telemetry memory, atomic routing, controller
  lifetime, NVML identity matching, overload accounting, and benchmark timing.
  Fixed WSL's versioned NVML library discovery and a ready-future collection
  iterator invalidation before the final measurements. No unresolved findings.

## Local Laya controller — 2026-09-25

The user replaced hosted Jev with local Laya. Evaluation uses the same Razer,
GCC 13.3 Release flags, CUDA 13.2 (`sm_86`), 100,000 × 384 dataset, seed 42,
two runtime CPU workers, queue capacity 1,024, and GPU batches up to 32 with a
2 ms wait. The existing 20-second trace offers 8,040 requests. Both adaptive
controllers use a 2-second interval for this comparison (Phase 3 used 500 ms).
Percentiles describe successful requests; rejected requests also count against
the 15 ms offered-latency SLO. These are single trace runs, not confidence intervals.

Inference uses Laya 0.3.20, PyTorch 2.14.0+cpu, Transformers 5.17.0, and the English
`convaiinnovations/laya` checkpoint at
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. The local service is warmed with three
requests before the load run; each native backend also receives three warmup
batches. The service remains resident but idle for the static/heuristic modes.
Its CPU work during Laya mode is included in whole-system utilization and competes
with native CPU work. The runtime never waits for the model on a request thread.
The default gate is 0.70 maximum class probability, without domain calibration.

### Inference budget experiment

The first full comparison used two CPU inference threads and an 800 ms timeout.
All 11 control calls timed out and fell back; no model decision was applied.
That run completed 6,037 requests, rejected 2,003, and had 4,607.29 ms p99, versus
6,919 completed / 1,121 rejected / 4,233.29 ms p99 for the 2-second heuristic.
Static GPU batching completed all 8,040 at 48.94 ms p99. The two-thread model run
is a real timeout/fallback measurement, not evidence of useful model scheduling.

To test whether CPU allocation caused those timeouts, the same 237-token request
was measured without runtime load: two warmups and five timed calls per thread
count, one loaded checkpoint, one inter-op thread. The policy and probability
were unchanged (`gpu_batch`, 0.697) across these configurations.

| Inference threads | Mean ms | Minimum ms | Maximum ms |
| ---: | ---: | ---: | ---: |
| 2 | 951.92 | 908.82 | 1,033.19 |
| 4 | 526.97 | 522.96 | 533.71 |
| 8 | 654.15 | 592.36 | 714.95 |

Four intra-op threads were selected based on this measurement; the timeout,
confidence gate, trace, and control interval were retained. No prompt or benchmark
was tuned to force a model win. The initial two-thread run remains in
`benchmark-output/phase4-laya/`; the final comparison is recorded separately.

### Four-thread comparison

| Policy | Completed / rejected | QPS | p50 ms | p95 ms | p99 ms | SLO violations | Changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu_latency | 3415 / 4625 | 170.74 | 3313.99 | 3933.07 | 3942.43 | 7967 | 0 |
| gpu_immediate | 8040 / 0 | 401.98 | 446.28 | 788.43 | 820.66 | 7332 | 0 |
| gpu_batch | 8040 / 0 | 401.99 | 25.40 | 47.90 | 55.45 | 7208 | 0 |
| heuristic | 6918 / 1122 | 345.89 | 74.39 | 3795.52 | 4178.30 | 7205 | 3 |
| laya | 6261 / 1779 | 313.04 | 131.96 | 4618.55 | 4726.93 | 7218 | 3 |

All modes had zero search failures and zero native controller exceptions. Only
static GPU modes accepted the full trace; their result checksums matched.
Laya made 11 bounded calls: nine returned a valid but below-threshold probability
and two timed out. All 11 applied the heuristic fallback. Mean call latency was
622.46 ms, maximum 802.00 ms (the configured timeout plus scheduling overhead).
Call counts include the final in-flight control call joined during shutdown;
throughput ends when accepted search work finishes, excluding that final join.

| Policy | Mean whole-system CPU % | Mean GPU % |
| --- | ---: | ---: |
| cpu_latency | 6.73 | 1.10 |
| gpu_immediate | 2.30 | 27.88 |
| gpu_batch | 2.33 | 29.45 |
| heuristic | 5.62 | 20.62 |
| laya | 13.75 | 20.00 |

Utilization is an arithmetic mean of 500 ms telemetry rows excluding the final
drained row and absent readings. Adaptive samples repeat between 2-second ticks;
these means are descriptive, not process CPU accounting or exact time integrals.
Laya's measured CPU cost and delayed fallback make it less useful than the native
heuristic here. Static GPU batching is strongest on this GPU-friendly trace.
Phase 3's faster 500 ms heuristic also performs better than the 2-second baseline;
slower control allows CPU backlog to form before routing changes.

The compact measured comparison is committed in `benchmarks/results/phase4.json`.
Raw summaries, trace/hash, commands, compiler metadata, Python package versions,
model revision, server log, and telemetry are in ignored
`benchmark-output/phase4-laya-4/`. The server used roughly 2 GiB RSS after warmup.

### Ungated diagnostic

A separate run set `--laya-min-confidence 0` to observe raw choices. This is not the
shipped confidence gate. All ten responses were accepted (mean 552.65 ms,
maximum 638.08 ms); Laya selected `gpu_batch` on every call, including sparse
traffic. Only the seven startup requests ran on CPU. All 8,040 queries completed,
with 401.98 QPS, p50 25.71 ms, p95 44.33 ms, p99 51.49 ms, 7,201 SLO violations,
and no rejections or errors. The checksum matched the static GPU runs.

This validates real model decisions reaching the C++ scheduler. It behaves like
static GPU batching on this trace; the small p99 difference between separate
runs is not evidence of a model speedup or useful adaptation. The default gated
mode remains experimental. Better domain calibration/training would be separate
work, outside this project's scope. Diagnostic artifacts are in
`benchmark-output/phase4-laya-diagnostic/`.

### Failure handling and review

The real server rejects oversized token inputs with HTTP 422. With the server
stopped, the 1,000-vector trace completes all 8,040 requests through heuristic
fallback (11 connection failures, no search failures). C++ tests also cover HTTP
errors, redirects, oversized/deep/malformed responses, unknown policies,
inconsistent probabilities, low confidence, unavailable GPU, memory guards,
control rate limiting, and search completion while the control call is blocked.
The independent Phase 4 review found no actionable regressions; its read-only
sandbox could not run HTTP/live tests, which were run separately on the Razer.

All ten Razer CUDA Release tests pass. Linux CPU ASan, UBSan, and TSan suites pass
(including AVX2 and Laya/HTTP tests; GPU tests explicitly skip in sanitizer builds).
TSan uses per-process `setarch x86_64 -R`. Mac Release and UBSan suites pass on
available backends. The default build with Laya disabled also builds and passes.
Prior Mac ASan startup and WDDM Compute Sanitizer limitations remain as documented.

## Browser dashboard and real contention — 2026-09-25

The final demo uses the same Razer / WSL2 hardware, GCC 13.3 Release build,
CUDA 13.2, 100,000 × 384 dataset, top-K 10, seed 42, two CPU workers, queue capacity
1,024, and batches up to 32 / 2 ms. The native heuristic samples every 500 ms.
Three warmup batches per backend precede server startup. Chrome on the Mac reads
HTTP/SSE through an SSH loopback tunnel; there is no simulated telemetry.

A browser-driven sequence captured 76 snapshots across quiet traffic (8 seconds),
burst traffic (10 seconds), burst plus real CUDA contention (10 seconds), and
quiet traffic with contention off (10 seconds). The competing workload launched
1,970 kernels. Observed policies were CPU latency → GPU batching → balanced →
GPU batching → CPU latency. The brief second batching phase drained backlog.

| Stage | Observed behavior |
| --- | --- |
| Quiet | CPU policy; all completions on CPU; queue empty |
| Burst | GPU batching; all completions on GPU by the end; queue peak 474 |
| Burst + contention | Measured GPU use reached 100%; balanced policy; final one-second completions split 46% CPU / 54% GPU; queue reached its 1,024 limit |
| Return to quiet | Queue drained; CPU policy and 100% CPU completions returned; final rolling p99 upper bound 7.78 ms |

The demo is a behavior/lifecycle check, not a scheduler performance comparison.
Overload produced 6,748 rejected queries, zero search failures, and a peak observed
rolling p99 upper bound of 3,152 ms. These costs remain visible in the UI. It does
not claim a 15 ms SLO under overload or reproduce the illustrative CPU percentages
in the spec. After the scripted sequence, quiet traffic continued; shutdown later
reported all 19,545 accepted requests completed with zero failures. Raw snapshots
are in ignored `benchmark-output/dashboard/demo.json`; the committed
[dashboard screenshot](dashboard.png) captures the contention stage.

A separate real-model browser smoke check confirmed that the Laya panel displays
actual maximum class probability, inference latency, and heuristic fallback.
The captured response had probability 0.4336, 795 ms latency, and `low_confidence`
status. All 16 decisions by that snapshot had fallen back (seven transport/model
errors), while 310 search requests had completed without failure. This checks
integration only; the controlled Phase 4 comparison remains the model evaluation.
The model process was stopped after the check, and the default heuristic retained.

### Final validation and review

- All 12 Razer CUDA Release CTest tests pass without skips, including backend
  equivalence, concurrent runtime, local-model failures, and real GPU contention.
- Linux CPU ASan, UBSan, and TSan suites pass with both dashboard and Laya enabled:
  nine executable tests pass, with the two CUDA tests explicitly skipped. This
  includes the new HTTP/SSE/control/shutdown integration check. TSan uses the
  previously documented per-process `setarch -R` workaround.
- Mac CPU Release suites pass with the dashboard enabled, both with and without
  Laya. AVX2/CUDA tests explicitly skip on ARM. JavaScript syntax checks pass.
- Chrome checks pass at desktop and 390-pixel phone widths: controls, custom
  startup top-K 20, table data, chart sizing, no horizontal page overflow, disabled
  CUDA controls in CPU mode, disconnect/restart recovery, and stream-limit recovery.
  Cached-page handlers are tested by dispatching the browser's persisted
  `pagehide` / `pageshow` events; this does not claim full BFCache eligibility.
- The real GPU demo verifies the contention switch in both directions, actual
  kernel execution, utilization-driven policy changes, and return to quiet CPU
  operation. Browser checks report no JavaScript exceptions.
- The independent final `codex review --uncommitted` identified three frontend
  issues: terminal EventSource errors did not retry, cached-page restoration did
  not reconnect, and valid custom startup top-K was missing from the selector.
  All three are fixed and covered by the browser regression check. A decorative
  switch element that intercepted native checkbox clicks was also fixed and
  verified in the real GPU demo. No unresolved review findings remain.

The review sandbox could not run socket-based integration checks; those ran
separately on both hosts. CPU sanitizers do not validate CUDA device memory;
the existing WDDM Compute Sanitizer and Mac ASan limitations remain unchanged.
