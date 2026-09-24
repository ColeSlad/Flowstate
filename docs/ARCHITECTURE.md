# Architecture

## Compute backends

`Dataset` owns a contiguous row-major float32 array. It validates shape and finite
values, exposes read-only spans, and is shared with each backend through
`shared_ptr<const Dataset>`. Explicit mt19937 integer-to-float mapping makes the
same seed deterministic across standard library implementations.

A `Query` owns its values and top-K. A `SearchResult` owns ordered IDs and scores.
The small `Backend` interface offers synchronous search and batch search. CPU
batch search loops over queries. Empty batches return no results. Validation
errors throw exceptions; no backend silently changes the requested algorithm.

Scalar and AVX2 implementations compute each row's dot product and maintain a
bounded top-K heap. Exact score ties use ascending IDs. Approximate comparisons
are confined to tests; putting a tolerance into the sorting comparator would
break strict weak ordering. Tests compare actual scores and ranked IDs against
an independent full sort using double-precision dot products, allowing differing
IDs only when reference scores tie within tolerance.

The AVX2 function uses unaligned loads, multiply/add, and a scalar tail. Its target
attribute isolates AVX instructions from the portable executable baseline.
Runtime CPU/OS feature detection gates its factory. Fused contraction is disabled
for the compute core to reduce avoidable differences between reduction orders.

## CUDA implementation (validated on RTX 3080 Ti Laptop GPU)

A CUDA instance owns the persistent dataset on device 0, one nonblocking stream,
four timing events, grow-on-demand device query/score buffers, and reusable host
buffers. RAII wrappers release device allocations, events, and the stream.
Dataset upload occurs once at construction.

Each search validates every query, flattens the batch, uploads queries, launches
one block per `(query, dataset row)`, reduces dot products in shared memory,
downloads scores, synchronizes, and performs exact top-K selection on the CPU.
The initial algorithm prioritizes correctness; it does not implement GPU top-K,
pinned host memory, or multiple streams. Larger batches require
`batch_size × dataset_size` score storage on both host and device.

CUDA API calls and launch/completion failures propagate as exceptions. The error
path synchronizes before host buffers can be reused; destruction also waits for
pending stream work. Destructors do not throw. Grid limits and allocation-size
arithmetic are checked before launch. Event intervals report upload, kernel, and
download separately; total wall-clock latency also includes host work. Pageable
host memory can introduce transfer staging/serialization, so those event
intervals should not be interpreted as isolated link bandwidth measurements.

## Runtime ownership and request lifecycle

`Runtime` owns fixed CPU workers and, when enabled, one GPU worker. CPU backend
instances are reentrant because searches mutate only local data. Only the GPU
worker calls the CUDA instance, preserving exclusive ownership of reusable
buffers, events, and its stream. Dataset/backend lifetimes exceed worker lifetimes.

Submission validates the query, assigns a request ID, and places an owning job
and promise in the chosen queue. One mutex protects both queues and counters;
separate condition variables wake CPU and GPU workers. The configured capacity
bounds total waiting work across both queues, excluding at most `cpu_workers`
CPU requests and `max_batch_size` GPU requests in flight. Queue-full and shutdown
rejections are synchronous exceptions. Accepted backend failures are delivered
through futures, and the worker remains available for subsequent work.

The GPU queue preserves FIFO order. A batch flushes at maximum size or when its
oldest request reaches the configured timeout. An immediate request behind a
partial batch flushes that batch without waiting; it then executes alone.
Shutdown stops admission, wakes all workers, drains accepted jobs (flushing any
partial batch immediately), and joins. A separate shutdown mutex serializes
concurrent join callers. No thread is detached. Results own their backend name
and data, and futures can outlive the runtime.

Completions record queue wait, backend execution, and submission-to-completion
latency, including result publication overhead up to the timestamp immediately
before promise fulfillment. CUDA event timings describe the whole batch and
must not be summed once per request. A mutex-protected snapshot reports queue
depths, in-flight work, successes/failures, rejections, flush counts, and cumulative
timings; device intervals are counted once per batch. Unavailable device timings
remain absent. No telemetry reader accesses CUDA resources directly.

## Control-plane boundary

Phase 3 adds an atomic policy to `Runtime`: CPU, immediate GPU, GPU batch, or a
50/50 alternating CPU/immediate-GPU split. A submission reads one policy and
stores a concrete route; policy changes never migrate queued or in-flight jobs.
Explicit route submission remains available. CPU always uses AVX2 when supported.

`HeuristicController` owns one joinable control thread. Every 500–2,000 ms it
reads a bounded runtime summary and samples system counters, then stores one
policy. Default precedence is: unavailable GPU or measured free memory below
64 MiB → CPU; rate ≤25 QPS with queue <16 → CPU; measured GPU use ≥90% → balanced;
queue ≥16 or rate ≥300 → GPU batch; rate ≥80 or measured CPU use ≥50% → balanced;
otherwise immediate GPU. Thresholds live in `HeuristicConfig`. Absent system
metrics are skipped, not filled with fabricated values. This is a transparent
baseline, not an optimized scheduler or evidence that adaptive routing always wins.

`RollingTelemetry` uses 300 reusable 100 ms buckets, about 2.5 MB per runtime.
Each contains counters and four fixed 256-bin logarithmic latency histograms.
No request history grows with traffic. Arrival counters include valid queue-full
attempts; completion counters include failures, while latency/throughput use
successful completions. The 1/5/30-second windows divide rates by their full
window length, including empty startup history. Boundary resolution is 100 ms;
percentiles round upward by at most 10%, with a 1 microsecond first bin and an
overflow bin. Snapshot aggregation shares the queue mutex, so sampling has a
small synchronization cost that is included in load measurements.

System sampling runs only in the control plane. Linux CPU use comes from deltas
in `/proc/stat`. Optional NVML resolves CUDA device 0 by PCI identity, then asks
for utilization and free device memory. Unsupported calls leave optional fields
empty. NVML availability is a build-time capability; its readings are neither
required for CPU-only execution nor substituted with synthetic values. See the
[NVIDIA device query reference](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html).
Controller snapshots include their sampling time, transitions, and errors. Stop
and destroy the controller before the runtime it references.

The load generator schedules bounded CSV phases independently of completions,
collects ready futures without waiting on the oldest one, and records queue-full
rejections. It retains a bounded maximum of two million latency samples for exact
benchmark percentiles. Producer lateness is measured against each burst's intended
arrival time; offered-latency/SLO reporting includes it. Queries have a fixed
vector dimension per dataset, while traces can change top-K. No synthetic GPU
contention or fabricated utilization is used.
Phase 4 will run Jev periodically over bounded summaries, with timeouts and
heuristic fallback; no Jev call will execute or block individual queries.
Phase 5 will expose telemetry to a dashboard independently of runtime correctness.
