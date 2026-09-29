# Progress

## Current Phase
All five phases complete. Repeated study and standalone recorded demo complete.

## Completed
- [x] Phase 1 — Compute backends
- [x] Phase 2 — Runtime
- [x] Phase 3 — Adaptive scheduling
- [x] Phase 4 — Local Laya integration
- [x] Phase 5 — Dashboard and polish

## Current Notes
- Standalone demo: static `web/` hosts the 76 unchanged GPU capture snapshots with explicit recorded labeling, play/pause, seeking, stage navigation, and source/report links. The native server selects live mode explicitly. Vercel settings and local preview instructions are included. Data equality/provenance, all-frame playback, keyboard/lifecycle/error handling, no backend requests, desktop/phone/dark appearance, and live-browser regressions pass. Mac Release build and all 10 available CTests pass (3 unavailable AVX2/CUDA tests skip). No compute or scheduler changes; no new performance claims. Ready to import on Vercel; no public deployment has been created.
- Repeated benchmark report (2026-09-28–29 UTC): all 300 trials complete, with 87.3M offered requests, 60.2M completed searches, and zero search/controller errors. Five shuffled repetitions cover 1K/100K/1M vectors and four policies; all 60 dynamic cases verify real CUDA contention. At 100K, GPU batching passes 1,000 offered QPS with p99 7.97–14.84 ms and zero rejections in every trial. All 15 Razer CUDA tests pass; Mac tests and the 27-trial tooling smoke run pass with unavailable backends skipped. Full artifact/hash/accounting checks and chart review pass. Compact data, three PNG/SVG charts, compiler metadata, limitations, and interrupted-run history are in `docs/benchmarks/2026-09-28/report.md`; reproduction is in `docs/BENCHMARKING.md`. Runtime policies and backends were unchanged.
- UI refresh (2026-09-27): simplified the dashboard into one primary panel, shortened labels, and moved secondary measurements/history into Runtime details. README screenshot uses real local CPU telemetry; the Razer was unreachable for a fresh GPU capture. Build, dashboard HTTP test, browser regressions, and desktop/phone/keyboard appearance checks pass.
- Canonical origin is `git@github.com:ColeSlad/Flowstate.git`. The source spec records the user-approved Jev → local Laya amendment.
- Scalar/AVX2/CUDA equivalence, bounded concurrency, microbatching, telemetry, and routing are validated on the Razer i9-12900H / RTX 3080 Ti Laptop under WSL2.
- Original five-phase validation: all 12 CUDA Release tests pass without skips. Linux CPU ASan/UBSan/TSan pass with dashboard and Laya enabled; Mac CPU Release passes with Laya on/off. Prior Mac ASan and WDDM Compute Sanitizer limits remain documented.
- Desktop/phone browser checks pass, including control validation, server restart, stream-limit recovery, cached-page lifecycle handlers, and custom startup top-K. Three final-review findings are fixed and regression-tested.
- The real GPU demo shows CPU → GPU batching → balanced → CPU, with a brief batching phase while draining. Measured GPU contention reaches 100%; overload rejections remain visible. Live local-model confidence/fallback display is also validated.
- Default-gated Laya underperformed the heuristic and fell back on all 11 evaluation decisions. Ungated Laya always chose GPU batching and performed similarly to static batching. The native heuristic remains the demo default; static batching won the recorded trace.
- Build/run instructions, demo sequence/screenshot, complete results, and limitations are in `README.md` and `docs/`. No remaining spec blockers or unresolved review findings.
