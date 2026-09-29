# Repeated benchmark report

This workflow measures the existing runtime. It does not change scheduler
thresholds, backends, or model behavior. The default comparison is static CPU,
GPU immediate, GPU batching, and the native 500 ms heuristic. Laya's separate
evaluation remains in [RESULTS.md](RESULTS.md).

The [completed 300-trial Razer study](benchmarks/2026-09-28/report.md) includes
all measured cases, individual summaries with UTC timestamps, compiler/build
metadata, the plotted burst telemetry, and PNG/SVG figures. It offered 87,318,000
requests with zero search/controller errors. Raw per-trial folders remain under
ignored `benchmark-output/`; the committed extract is about 1.3 MB.

## Run on the configured Razer

From Ubuntu 24.04 in WSL, in the repository:

```sh
export PATH=/usr/local/cuda-13.2/bin:/usr/lib/wsl/lib:$PATH
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DFLOWSTATE_ENABLE_CUDA=ON
cmake --build build-gpu -j4
ctest --test-dir build-gpu --output-on-failure
python3 -m venv .venv-bench
.venv-bench/bin/python -m pip install -r benchmarks/requirements.txt
.venv-bench/bin/python benchmarks/run_report.py --binary build-gpu/flowstate_load \
  --cuda on --cpu-backend avx2 --label 'i9-12900H; RTX 3080 Ti Laptop; WSL2' \
  --output benchmark-output/report
```

Keep the laptop plugged in and awake. Stop the Flowstate dashboard/load generator
and local model service before measuring. Record other workloads or unusual
conditions in `--label`. The script records hardware/toolchain details but does
not change power settings, lock clocks, pin CPUs, or stop other processes.
Use a clean, committed source revision for published performance results.

The default is **300 serial trials** with about **160 minutes of scheduled
traffic**, plus dataset construction, warmup, and final drain. Do not run benchmark
processes in parallel. Python 3.11+ is required. Matplotlib is used only to render
figures; a remote host can use `--no-plots` and its output can be copied to a host
with Matplotlib for rendering:

```sh
.venv-bench/bin/python benchmarks/run_report.py --output benchmark-output/report --render-only
```

## Workloads and ordering

All searches use float32 dot products, dimension 384, top-K 10, and seed 42.
Each run cycles through the same 32 generated queries. Dataset and query seeds
are independent. Defaults are two CPU workers, queue capacity 1,024, GPU batch
maximum 32, and oldest-request wait 2 ms. The configuration is saved verbatim.

| Dataset vectors | Steady offered QPS grid | Burst / contention peak QPS |
| ---: | --- | ---: |
| 1,000 | 1,000 / 10,000 / 30,000 / 60,000 | 60,000 |
| 100,000 | 100 / 500 / 1,000 / 1,500 | 1,500 |
| 1,000,000 | 10 / 50 / 100 / 150 | 150 |

Each steady cell lasts 30 seconds with individual evenly scheduled arrivals.
The dynamic trace has four 10-second phases: quiet at 10 QPS, burst traffic in
groups of 32, the same burst traffic with a real competing CUDA kernel, then
quiet with contention off. That kernel is the existing dashboard competitor;
it runs on the same GPU and its actual completed kernel count is recorded.
The capacity table applies **only to steady traffic without injected contention**.

Five repetitions visit all cases once per repetition, shuffling both cases and
policies with a recorded order seed. This reduces fixed-order bias; it does not
eliminate thermal drift or outside workload noise. Each trial is a fresh process
and warms both actual backend instances with three batches before starting an
empty runtime. Setup and persistent dataset upload are excluded. Controller
startup, routing lag, rejected work, and final drain remain visible.

To change the grid, supply `--vectors`, `--rates`, `--repetitions`,
`--duration-seconds`, and/or `--phase-seconds`. `--rates` applies the same list to
each requested size. An input trace is limited to two million requests; the
runner rejects an oversized plan before launching it. Increasing duration may
require lowering the largest rates. A slower system may need lower rates to
locate a passing cell; report the expanded grid rather than discard old failures.

## Capacity criteria

The report calls a rate passing only when **every planned repetition** meets:

- Successful-request **offered p99 ≤15 ms**, including producer lateness.
- Rejected / offered requests **≤0.1%**.
- All SLO violations / offered requests **≤1%**. Violations include late
  completions and every rejection/failure.
- At least **1,000 successful samples per trial**.
- Zero search failures and controller exceptions.
- Measured elapsed time including drain **≤105% of the scheduled duration**.

It requires at least three repetitions and completion of the full rate grid for
that dataset/policy before publishing its highest passing rate. Defaults use
five repetitions. These thresholds are recorded and configurable, not silently
relaxed to obtain a result. A low-rate trial with too few samples remains in the
table but cannot establish capacity. For example, 10 QPS for 30 seconds provides
only 300 samples.

The headline is **highest tested passing offered QPS**. It is not an interpolated
maximum or a guarantee of indefinite operation. If the highest grid point passes,
capacity is not bounded by the experiment. If none passes, the report says so;
it does not claim the system has zero capacity. Non-monotonic trial outcomes are
retained. A finite test cannot establish production tail latency, and 1,000
samples is only a minimum guard, not a precise p99 confidence guarantee.

## Artifacts and charts

`report.md` includes all measured cells, sample counts in the linked data, pass
counts, median throughput/p99 and observed min–max variation. Standard deviations
are in `report.json`; these are variation across trials, not pooled percentiles
or confidence intervals. Failed/rejected work is shown alongside successful
latency so dropping requests cannot masquerade as an improvement.

Three figures are exported as PNG and SVG:

1. `capacity`: highest tested passing rate versus dataset size. Missing claims
   are labeled instead of plotted as zero capacity.
2. `latency`: offered p99 versus offered QPS, with median/min–max error bars and
   markers for cells failing the full capacity criteria.
3. `burst`: completions, rolling p99, queue depth, and cumulative rejections
   through the changing workload. Always uses trial 1 at 100K vectors if present,
   otherwise the middle dataset size. Shading marks the requested contention
   window; kernel counters verify execution. It never selects the best-looking run.

The burst plot uses the runtime's one-second histogram p99 **upper bound** and
500 ms telemetry, not the summary's exact quantile. Controller samples can be
older than the telemetry row; `sample_age_ms` records that. Producer collection
can lag worker completion slightly; cumulative completion counts describe
collected futures. Summary timing ends after accepted search work drains,
excluding controller shutdown and contention-thread cleanup.

`manifest.json` records the shuffled plan, full configuration, source commit and
working-tree status, source/binary SHA-256 hashes, compiler commands/flags,
Python version, CPU, GPU, driver, and CUDA version when available. Each trial
retains its command, trace, stdout CSV, stderr, telemetry, start/end timestamps,
and artifact hashes. `trials.csv` contains individual summaries. `plot_metadata.json`
records Matplotlib's version. Raw artifacts stay under ignored `benchmark-output/`;
publish only a compact report/data extract and figures after reviewing the results.

## Interruptions and verification

Repeat the exact command with `--resume`. Completed trials are checked by hash
and skipped; an interrupted trial is rerun. Resume rejects changed host, source,
binary, or settings. Keep using the same revision until the run finishes. A new
binary requires a new output directory. `--render-only` reads saved measurements
without requiring the original host or binary; incomplete reports suppress the
affected capacity claims.

Search/controller failures or invalid accounting stop the run and preserve its
artifacts. Fully completed equivalent workloads must have matching ID checksums.
Checksums supplement the backend equivalence tests; they are not a replacement
for them. Rejection is an expected overload outcome and does not abort a trial.

For a short **CPU-only tooling check**, including charts:

```sh
.venv-bench/bin/python benchmarks/run_report.py --binary build-demo/flowstate_load \
  --cuda off --cpu-backend scalar --modes cpu_latency --vectors 1000 100000 1000000 \
  --rates 50 100 --repetitions 3 --duration-seconds 5 --phase-seconds 2 \
  --label 'CPU-only short tooling validation; not GPU performance evidence' \
  --output benchmark-output/report-smoke
```

Short checks intentionally fail the minimum sample-count criteria. CPU-only mode
omits contention rather than simulating it. CTest includes statistics regression
checks and, for Release builds with Python available, native report/resume tests.
CUDA builds also test real contention through this benchmark path.
