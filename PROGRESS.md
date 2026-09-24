# Progress

## Current Phase
Phase 3 — Adaptive scheduling

## Completed
- [x] Phase 1 — Compute backends
- [x] Phase 2 — Runtime
- [ ] Phase 3 — Adaptive scheduling
- [ ] Phase 4 — Jev integration
- [ ] Phase 5 — Dashboard and polish

## Current Notes
- `agents/FLOWSTATE_CODEX_LEAN_SPEC.md` is the source of truth; canonical origin verified.
- Phase 2 implements fixed CPU workers, bounded CPU/GPU queues, futures, request timing, and GPU size/timeout batching.
- Remote environment: i9-12900H, RTX 3080 Ti Laptop 16 GB, WSL 2.7.14 / Ubuntu 24.04, GCC 13.3, CUDA 13.2.86.
- All seven CUDA Release tests and Linux ASan/UBSan/TSan CPU checks pass. TSan uses per-process `setarch -R`; Compute Sanitizer's WDDM limitation remains documented.
- Crossover at 1k × 384: batch 1 AVX2 0.028 ms vs CUDA 0.123 ms; batch 8 AVX2 0.241 ms vs CUDA 0.166 ms.
- Full 45-row matrix and transfer/kernel timings are summarized in `docs/RESULTS.md`; raw output stays ignored.
- Phase 1 review found no unresolved correctness issues; hardware validation needed no implementation changes.
- Runtime tests cover concurrent submission, queue capacity, shutdown under load, batching, and backend failure recovery.
- Deeper Phase 2 review found completion latency stopped too early; fixed and regression-tested on both hosts. No unresolved findings.
- Runtime batching vs immediate GPU: 8.49× throughput at 1k × 384; 1.21× at 100k × 384 (512-request bursts). Full baselines and tails are in `docs/RESULTS.md`.
- Next: rolling telemetry, repeatable load phases, and native heuristic policy selection.
