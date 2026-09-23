# Flowstate — Codex Build Specification

## 0. Goal

Build **Flowstate**, a self-optimizing heterogeneous C++/CUDA runtime for vector similarity search.

Flowstate should dynamically route vector-search workloads across:

- scalar CPU
- AVX2 CPU
- CUDA GPU
- CUDA GPU with microbatching

The runtime should observe live system state and adapt execution policy.

A Jev-based controller will later act as a **slow control-plane policy selector**.

Jev must **not** run in the per-request hot path.

The core of the project must remain:

- low-level C++
- concurrency
- SIMD
- CUDA
- runtime scheduling
- batching
- telemetry
- performance engineering

The final result should be a polished systems project with a strong browser demo.

---

# 1. GitHub Repository

Canonical repository:

```text
git@github.com:ColeSlad/Flowstate.git
```

At the beginning of work, inspect:

```bash
git status
git branch --show-current
git remote -v
```

Expected remote:

```text
origin  git@github.com:ColeSlad/Flowstate.git
```

If `origin` is missing:

```bash
git remote add origin git@github.com:ColeSlad/Flowstate.git
```

If `origin` exists but points somewhere else, stop and report the mismatch.

Do not silently replace an existing remote.

Never use force push unless explicitly instructed.

Do not commit:

- `.env`
- API keys
- credentials
- build directories
- secrets
- large generated benchmark artifacts

---

# 2. Project Summary

Flowstate performs exact vector similarity search over an in-memory dataset.

Each query asks for the top-K vectors with the highest dot-product similarity.

The same operation can execute through multiple backends.

```text
Incoming query
      |
      v
+-----------------------------+
|        Flowstate Runtime    |
|                             |
| queue -> scheduler          |
|             |               |
|             +--> AVX2 CPU   |
|             +--> CUDA GPU   |
|             +--> GPU batch  |
|                             |
| telemetry -> policy control |
+-----------------------------+
                 |
                 v
              Jev
       high-level policy only
```

The runtime should collect telemetry such as:

- queue depth
- request arrival rate
- backend latency
- backend throughput
- CPU utilization
- GPU utilization
- GPU memory availability
- GPU batch size
- pending GPU work

It should dynamically select among a small set of policies.

Example:

```cpp
enum class Policy {
    CpuLatency,
    GpuImmediate,
    GpuBatch,
    Balanced
};
```

---

# 3. Control Plane vs Data Plane

This separation is mandatory.

## Data plane

The native C++ runtime handles:

- request submission
- queueing
- worker threads
- batching
- synchronization
- backend execution
- result completion
- telemetry collection

The data plane must remain local and fast.

## Control plane

The policy controller runs periodically.

Recommended interval:

```text
500 ms – 2000 ms
```

It observes summarized telemetry and selects a high-level execution policy.

Jev must never be called for each request.

Jev must never directly execute work.

The C++ runtime is always responsible for actual scheduling.

---

# 4. Core Workload

Use exact vector similarity search over generated in-memory vectors.

Recommended defaults:

```text
dataset size: 100,000
dimension:    384
type:         float32
metric:       dot product
top-K:        10
seed:         42
```

Allow reasonable CLI configuration.

Example:

```bash
./flowstate_server \
  --vectors 100000 \
  --dimension 384 \
  --top-k 10 \
  --seed 42
```

The dataset must be deterministic when given the same seed.

---

# 5. Correctness Requirements

All execution backends must return equivalent results within floating-point tolerance.

For a fixed dataset and query:

```text
scalar
AVX2
CUDA
```

must return the same vector IDs in the same order unless scores are tied within tolerance.

Tests should cover:

- normal queries
- top-K = 1
- small datasets
- dimensions not divisible by 8
- invalid dimensions
- top-K larger than dataset
- deterministic dataset generation
- backend equivalence

Correctness comes before optimization.

---

# 6. Strict Scope

## In scope

- C++20
- CMake
- scalar vector search
- AVX2 vector search
- CUDA vector search
- CUDA microbatching
- worker threads
- bounded request queues
- runtime telemetry
- heuristic policy selection
- Jev policy selection
- load generation
- benchmark tooling
- simple realtime browser dashboard

## Out of scope

Do not implement these unless the core project is already complete and polished:

- HNSW
- IVF
- approximate nearest neighbor indexes
- persistent vector database
- distributed execution
- multiple machines
- multiple GPUs
- custom allocator
- custom HTTP server
- custom TLS
- Kubernetes
- LLM inference engine
- model serving
- compiler/JIT
- operating system components
- custom RPC framework
- lock-free queue unless profiling proves it matters
- AVX-512 before AVX2 is complete
- hot code patching
- code generation
- automatic CUDA kernel synthesis
- large frontend framework unless necessary

Prefer a finished, measured system over extra features.

---

# 7. Suggested Repository Structure

Use approximately:

```text
Flowstate/
├── CMakeLists.txt
├── README.md
├── PROGRESS.md
├── .gitignore
│
├── docs/
│   ├── ARCHITECTURE.md
│   └── RESULTS.md
│
├── include/
│   └── flowstate/
│       ├── backend.hpp
│       ├── query.hpp
│       ├── runtime.hpp
│       ├── scheduler.hpp
│       ├── telemetry.hpp
│       └── policy.hpp
│
├── src/
│   ├── main.cpp
│   │
│   ├── backends/
│   │   ├── scalar.cpp
│   │   ├── avx2.cpp
│   │   └── cuda.cu
│   │
│   ├── runtime/
│   │   ├── runtime.cpp
│   │   ├── worker_pool.cpp
│   │   ├── scheduler.cpp
│   │   └── batcher.cpp
│   │
│   ├── telemetry/
│   │   ├── metrics.cpp
│   │   └── rolling_window.cpp
│   │
│   ├── policy/
│   │   ├── heuristic.cpp
│   │   └── jev.cpp
│   │
│   └── server/
│       └── telemetry_server.cpp
│
├── tests/
│
├── benchmarks/
│
└── web/
    ├── index.html
    ├── app.js
    └── style.css
```

Do not create unnecessary abstraction layers.

---

# 8. Coding Guidelines

Use C++20.

Prefer:

- RAII
- deterministic ownership
- stack allocation where reasonable
- `std::unique_ptr` for ownership
- `std::span` for non-owning contiguous data
- bounded queues
- explicit shutdown paths
- minimal global state
- allocation-light hot paths
- straightforward synchronization

Avoid:

- detached threads
- unbounded queues
- deep inheritance
- abstraction for abstraction's sake
- premature lock-free structures
- premature optimization

CUDA resources should use RAII-style wrappers where practical:

- device memory
- CUDA streams
- CUDA events
- pinned memory

---

# 9. Build Requirements

Use CMake.

Expected workflow:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CUDA should be optional at configuration time if practical.

Example:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLOWSTATE_ENABLE_CUDA=ON
```

If CUDA is unavailable, CPU-only targets should still build where reasonable.

---

# 10. Progress Tracking

Do not create a ticket system.

Use only:

```text
PROGRESS.md
```

Keep it small.

Suggested format:

```markdown
# Progress

## Current Phase
Phase 2 — Runtime

## Completed
- [x] Phase 1 — Compute backends
- [ ] Phase 2 — Runtime
- [ ] Phase 3 — Adaptive scheduling
- [ ] Phase 4 — Jev integration
- [ ] Phase 5 — Dashboard and polish

## Current Notes
- Scalar, AVX2, and CUDA return equivalent results.
- AVX2 currently wins for small batches.
- CUDA wins for larger batches.
```

Update `PROGRESS.md` when completing meaningful work.

Do not spend significant time maintaining project-management files.

---

# 11. Execution Workflow

Work through the phases in order.

For each phase:

1. inspect the current repository
2. implement the smallest coherent version
3. add or update tests
4. build
5. run tests
6. run relevant benchmark
7. inspect the diff for unnecessary complexity
8. fix obvious correctness or design issues
9. update `PROGRESS.md`
10. commit
11. push to GitHub
12. continue to the next phase automatically

Do not stop after every phase unless:

- blocked by missing credentials
- blocked by missing CUDA/tooling
- tests repeatedly fail
- the next step requires a major architecture decision
- the spec is ambiguous in a way that changes scope
- the Git remote is incorrect
- a destructive Git operation would be required

Otherwise, continue.

---

# 12. Git Workflow

Use milestone-sized commits.

Examples:

```text
chore: initialize flowstate project
feat: add vector search backends
feat: add concurrent runtime and gpu batching
feat: add adaptive scheduling and telemetry
feat: add jev policy controller
feat: add realtime flowstate dashboard
bench: add scheduler evaluation
```

Before committing:

```bash
git status
git diff
git diff --cached
```

Stage only files relevant to the completed work.

Then:

```bash
git commit -m "<message>"
git push -u origin HEAD
```

If upstream already exists:

```bash
git push
```

Never use:

```text
git push --force
git push --force-with-lease
git reset --hard
git clean -fd
```

unless explicitly instructed.

Do not rewrite published history.

---

# 13. Phase 1 — Compute Backends

Build the computational core first.

## 13.1 Project setup

Create:

- CMake project
- source directories
- test target
- benchmark target
- CUDA feature flag
- `README.md`
- `PROGRESS.md`

The project must build before continuing.

---

## 13.2 Scalar backend

Implement exact top-K dot-product search.

Example types:

```cpp
struct Query {
    std::vector<float> values;
    std::size_t top_k;
};

struct SearchResult {
    std::vector<std::size_t> ids;
    std::vector<float> scores;
};
```

Create a scalar backend.

Measure:

- average latency
- p50
- p95
- queries/sec

---

## 13.3 AVX2 backend

Implement dot product using AVX2 intrinsics.

Use:

- `_mm256_loadu_ps`
- `_mm256_mul_ps`
- `_mm256_add_ps`

Handle scalar tail elements safely.

Benchmark dimensions:

```text
128
384
768
```

Compare against scalar.

Do not add AVX-512.

---

## 13.4 CUDA backend

Implement CUDA search.

Start with correctness.

Required:

- persistent dataset on GPU
- query upload
- kernel launch
- result retrieval
- proper CUDA error handling
- clean resource destruction

Measure:

- single-query latency
- batch throughput
- host/device transfer cost
- kernel time

The phase is complete when:

- scalar works
- AVX2 works
- CUDA works
- results match
- benchmarks exist
- at least one CPU/GPU crossover is demonstrated

Example expected behavior:

```text
small workload:
AVX2 faster

large batch:
CUDA faster
```

Commit and push.

---

# 14. Phase 2 — Runtime

Build the execution runtime.

## 14.1 Backend interface

Use a minimal interface.

Example:

```cpp
class Backend {
public:
    virtual ~Backend() = default;
    virtual SearchResult search(const Query& query) = 0;
    virtual std::string_view name() const = 0;
};
```

Avoid over-engineering.

---

## 14.2 Worker pool

Implement:

- fixed worker threads
- bounded job queue
- clean shutdown
- request IDs
- timing
- async result completion

Example public API:

```cpp
std::future<SearchResult> submit(Query query);
```

Do not use detached threads.

---

## 14.3 GPU microbatcher

Requests targeting the GPU should be able to batch.

Flush when either:

```text
batch_size >= configured maximum
```

or:

```text
oldest_request_wait >= configured timeout
```

Suggested initial values:

```text
max_batch_size = 32
max_wait_us = 2000
```

Track:

- batch size
- queue wait
- kernel time
- end-to-end latency

The phase is complete when:

- concurrent requests work
- shutdown is safe
- GPU batching works
- timeout flush works
- size flush works
- batching improves throughput under at least one workload

Run a deeper review at the end of this phase because concurrency and CUDA synchronization are now involved.

Suggested review:

```bash
codex review --uncommitted
```

or review against the previous stable commit.

Prioritize:

- races
- lifetime bugs
- CUDA synchronization
- shutdown behavior
- queue correctness
- unnecessary abstractions

Fix meaningful findings, then commit and push.

---

# 15. Phase 3 — Adaptive Scheduling

Add telemetry, workload generation, and heuristic policy switching.

## 15.1 Telemetry

Track:

```cpp
struct RuntimeStats {
    double arrival_rate;

    double scalar_p50_ms;
    double scalar_p95_ms;
    double scalar_p99_ms;

    double simd_p50_ms;
    double simd_p95_ms;
    double simd_p99_ms;

    double cuda_p50_ms;
    double cuda_p95_ms;
    double cuda_p99_ms;

    std::uint64_t queue_depth;
    std::uint64_t pending_gpu_jobs;

    double cpu_utilization;
    double gpu_utilization;

    std::size_t gpu_memory_free_bytes;
};
```

Unavailable metrics should be represented as unavailable.

Do not fake telemetry.

Use rolling windows.

Suggested:

```text
1 sec
5 sec
30 sec
```

---

## 15.2 Load generator

Build repeatable workload phases.

Example:

```text
0–20 sec    low traffic
20–40 sec   CPU-heavy burst
40–60 sec   strong batching opportunity
60–80 sec   GPU contention
80–100 sec  low traffic
```

Support:

- fixed seed
- configurable request rate
- bursts
- query size variation
- JSON or CSV benchmark output

---

## 15.3 Heuristic controller

Create:

```cpp
enum class Policy {
    CpuLatency,
    GpuImmediate,
    GpuBatch,
    Balanced
};
```

Example behavior:

```text
GPU unavailable:
    CpuLatency

high queue depth + healthy GPU:
    GpuBatch

low load:
    CpuLatency

moderate CPU + GPU usage:
    Balanced
```

Keep thresholds centralized and configurable.

Do not build a generic rules engine.

Policies should map roughly to:

### CpuLatency

Prefer AVX2.

### GpuImmediate

Use GPU without intentional queueing delay.

### GpuBatch

Aggressively batch.

### Balanced

Split work between CPU and GPU.

The phase is complete when workload changes cause meaningful runtime policy transitions.

Commit and push.

---

# 16. Phase 4 — Jev Integration

Only add Jev after the heuristic system works end to end.

Create:

```cpp
class JevPolicyController;
```

Jev receives summarized runtime state.

Example:

```json
{
  "objective": "keep p99 latency under 15 ms while maximizing throughput",
  "queue_depth": 84,
  "arrival_rate": 8900,
  "cpu_utilization": 0.91,
  "gpu_utilization": 0.42,
  "simd_p95_ms": 11.4,
  "cuda_p95_ms": 6.2,
  "gpu_batch_p95_ms": 4.1
}
```

Jev may choose only:

```text
cpu_latency
gpu_immediate
gpu_batch
balanced
```

Expected control flow:

```text
RuntimeStats
     |
     v
serialize bounded state
     |
     v
Jev
     |
     v
typed policy choice
     |
     v
confidence check
     |
     +--> accepted
     |
     +--> heuristic fallback
```

Required safeguards:

- request timeout
- malformed response handling
- low-confidence fallback
- service-unavailable fallback
- no blocking in the request hot path
- rate limiting
- last known good policy where appropriate

Example:

```cpp
if (!decision.valid ||
    decision.confidence < config.min_confidence) {
    return heuristic.choose(stats);
}
```

Record:

- Jev latency
- selected policy
- confidence
- fallback count
- error count

API credentials must come from environment variables or secure config.

Never commit secrets.

---

# 17. Jev Evaluation

Compare at least:

```text
Static AVX2
Static CUDA
Heuristic
Jev
```

Use the same workload trace and seed.

Measure:

- throughput
- p50 latency
- p95 latency
- p99 latency
- SLO violations
- CPU utilization
- GPU utilization
- policy changes
- Jev fallbacks
- Jev control-plane latency

Do not tune the benchmark to force Jev to win.

If heuristic beats Jev, document that honestly.

The interesting question is:

> When does semantic/adaptive policy control help, and when is it unnecessary?

Run a deeper review after Jev integration.

Prioritize:

- failure handling
- hidden blocking
- bad fallback behavior
- control-plane/data-plane violations
- secrets handling
- unnecessary AI usage

Commit and push.

---

# 18. Phase 5 — Browser Dashboard

Build a lightweight frontend.

Prefer:

```text
HTML
CSS
vanilla JavaScript
```

Use a simple realtime transport such as:

- WebSocket
- Server-Sent Events

Do not introduce React unless necessary.

The runtime must not depend on the dashboard for correctness.

---

## 18.1 Dashboard information

Show:

```text
request rate
CPU utilization
GPU utilization
queue depth
current policy
selected backend distribution
p95 / p99 latency
Jev confidence
Jev fallback state
```

Visualize routing clearly.

Example:

```text
CPU   ██████████
GPU   ███████████████████
```

Show policy transitions:

```text
SIMD
  ↓
GPU_BATCH
  ↓
BALANCED
```

---

## 18.2 Demo controls

Only add a few controls:

1. traffic level
2. workload/query size
3. GPU contention

Avoid a huge configuration panel.

---

# 19. Demo Target

Optimize the entire project around a short demo.

## 0–10 seconds

Low traffic:

```text
CPU: 25%
GPU: 10%
Policy: CPU_LATENCY
Backend: AVX2
```

Explain:

> Flowstate chooses how work runs based on live machine conditions.

---

## 10–20 seconds

Increase traffic.

Show:

```text
CPU: 95%
queue depth increasing
```

Policy changes:

```text
CPU_LATENCY
     ↓
GPU_BATCH
```

Requests visibly shift to GPU.

---

## 20–30 seconds

Introduce GPU contention.

Show:

```text
GPU: 98%
```

Policy changes:

```text
GPU_BATCH
    ↓
BALANCED
```

Some work moves back to CPU.

---

## 30–40 seconds

Reduce traffic.

Policy returns toward AVX2.

---

## 40–50 seconds

Show a simple comparison:

```text
Static CPU
Static GPU
Heuristic
Jev
```

with:

```text
throughput
p99
SLO violations
```

A nontechnical viewer should understand:

> The runtime watches the machine and automatically changes the way work is executed.

---

# 20. GPU Contention

Create a controlled way to make GPU execution less attractive.

Prefer real contention where practical:

- competing CUDA kernel
- controlled GPU workload
- memory pressure

If artificial delay is used for demo/testing, label it clearly.

Do not present synthetic delay as real GPU telemetry.

---

# 21. Benchmark Discipline

Benchmark in Release mode.

Record:

- CPU model
- GPU model
- compiler
- compiler flags
- CUDA version
- dataset size
- dimension
- batch size
- worker count

Warm up before measuring.

Do not claim speedups without recording the baseline.

---

# 22. Testing Strategy

Unit tests should cover:

- dot product
- top-K
- deterministic dataset generation
- rolling metrics
- scheduler behavior
- policy parsing

Backend equivalence:

```text
Scalar == AVX2 == CUDA
```

within tolerance.

Concurrency tests:

- concurrent submissions
- queue capacity
- batch timeout
- batch size flush
- shutdown under load
- policy switching while work is in flight

Failure tests:

- CUDA unavailable
- Jev timeout
- Jev invalid response
- Jev low confidence
- queue full
- missing telemetry fields
- shutdown with pending work

---

# 23. Sanitizers and Profiling

Where compatible, support:

```text
AddressSanitizer
UndefinedBehaviorSanitizer
ThreadSanitizer
```

Use profiling tools only when useful.

Possible tools:

- `perf`
- Nsight Systems
- Nsight Compute
- compiler optimization reports

Do not optimize by intuition alone.

For meaningful optimization:

1. state hypothesis
2. benchmark baseline
3. implement
4. benchmark again
5. keep only if justified

---

# 24. Review Strategy

Do not run a heavyweight independent review after every small change.

Use deeper reviews at these boundaries:

```text
after Phase 2 — runtime/concurrency/CUDA
after Phase 4 — Jev integration
after Phase 5 — final project
```

Example:

```bash
codex review --uncommitted
```

or:

```bash
codex review --base main
```

Review priorities:

- correctness
- races
- memory lifetime
- CUDA synchronization
- shutdown behavior
- architectural violations
- hidden blocking
- benchmark mistakes
- scope creep
- unnecessary abstractions
- missing tests

Fix meaningful findings before pushing the phase.

---

# 25. README Requirements

The final `README.md` should include:

- project summary
- architecture diagram
- build instructions
- CUDA requirements
- how to run
- how to benchmark
- how to launch dashboard
- explanation of control plane vs data plane
- benchmark results
- demo GIF/video placeholder
- key technical challenges

Avoid marketing fluff.

Explain what was actually built.

---

# 26. Architecture Documentation

Maintain:

```text
docs/ARCHITECTURE.md
```

Keep it concise but useful.

It should explain:

- backend model
- thread model
- queue ownership
- request lifecycle
- GPU batching
- telemetry
- scheduler
- Jev control plane
- failure handling

Update only when architecture meaningfully changes.

---

# 27. Results Documentation

Maintain:

```text
docs/RESULTS.md
```

Include:

- benchmark setup
- hardware
- workloads
- scalar vs AVX2 vs CUDA
- batching results
- static vs heuristic vs Jev scheduling
- limitations
- unexpected results

Do not cherry-pick results.

---

# 28. Anti-Scope-Creep Rules

Before adding something, ask:

1. Does it improve the core runtime?
2. Does it improve the demo?
3. Does it improve the evaluation?
4. Is it justified by measurements?

If the answer is no to all four, do not build it.

The goal is:

```text
finished
polished
measured
understandable
technically deep
```

not:

```text
largest possible codebase
```

---

# 29. Definition of Done

Flowstate is complete when all of these work:

- scalar search
- AVX2 search
- CUDA search
- concurrent runtime
- GPU microbatching
- telemetry
- workload generator
- heuristic scheduling
- Jev policy controller
- safe fallback behavior
- reproducible evaluation
- realtime browser dashboard
- polished demo
- documented benchmark results

Do not delay completion for optional extensions.

---

# 30. Optional Extensions

Only after the core project is finished:

## Performance

- pinned host buffers
- multiple CUDA streams
- CPU affinity
- NUMA awareness
- improved data layout
- optimized GPU top-K

## Scheduling

- hysteresis
- SLO-aware scheduling
- adaptive batch timeout
- cost-aware routing
- online exploration

## Runtime

- cancellation
- priorities
- work stealing
- backend health checks

## AI systems

- learned latency predictor
- policy replay viewer
- compare Jev to a small local model
- confidence-aware exploration

---

# 31. Initial Codex Instructions

Read this file completely before changing the repository.

Then:

1. inspect Git state and remote
2. verify the GitHub repo is:
   `git@github.com:ColeSlad/Flowstate.git`
3. create or update `PROGRESS.md`
4. begin Phase 1
5. work through the phases in order
6. build and test continuously
7. benchmark where relevant
8. commit and push completed phases
9. continue automatically unless genuinely blocked

Do not create a ticket system.

Do not stop after every small milestone.

Do not ask for approval for ordinary implementation choices that are already covered by this spec.

Use the simplest implementation that satisfies the requirements.

Prioritize:

```text
correctness > cleverness
measurement > intuition
simple architecture > framework-building
finished demo > endless scope
systems depth > feature count
```

When the project is complete or blocked, report:

- what was implemented
- current phase
- tests run
- benchmark highlights
- review findings
- commit hash
- branch
- push status
- any remaining blockers
