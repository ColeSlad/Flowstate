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
- `agents/FLOWSTATE_CODEX_LEAN_SPEC.md` is the source of truth; canonical origin verified.
- Scalar, AVX2, and CUDA pass equivalence tests on the Razer; all five CUDA Release tests pass without skips.
- Remote environment: i9-12900H, RTX 3080 Ti Laptop 16 GB, WSL 2.7.14 / Ubuntu 24.04, GCC 13.3, CUDA 13.2.86.
- Linux ASan/UBSan CPU checks pass, including AVX2. Compute Sanitizer cannot initialize the WDDM debugger interface; documented separately.
- Crossover at 1k × 384: batch 1 AVX2 0.028 ms vs CUDA 0.123 ms; batch 8 AVX2 0.241 ms vs CUDA 0.166 ms.
- Full 45-row matrix and transfer/kernel timings are summarized in `docs/RESULTS.md`; raw output stays ignored.
- Phase 1 review found no unresolved correctness issues; hardware validation needed no implementation changes.
- Next: fixed CPU workers, bounded queues, futures/request timing, GPU size/timeout batching, shutdown tests, and runtime benchmark.
