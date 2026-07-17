# Development log

## 2026-07-17 — v0–v3 skeleton

- Added validated public/hidden state protocols and a strict JSON Action parser.
- Added GT-free reset/step, point/box dispatch, candidate-state rebuilding, Verifier
  feedback, finish handling, trajectory persistence, and history-best selection.
- Added pure NumPy instance extraction/matching as a deterministic adapter fallback.
- Added the modern Qwen3-VL-2B multimodal Adapter with explicit T1/T2/mask labels.
- Added SimpleClick and OmniOVCD/SAM3 adapter boundaries for isolated environments.
- Added a rule Verifier baseline, frozen-feature trainable head, and offline GT mask
  perturbation labels.
- Added CPU unit tests and a two-round synthetic smoke loop.

Validation performed with Python 3.10.20 / NumPy 1.26.4:

```text
12 unit tests passed
two-round trajectory smoke passed
SegAgent external prev_mask forwarding smoke passed
```

Not yet validated: real Qwen3-VL generation, real OmniOVCD initialization, real SAM3
box prompting, real SimpleClick refinement quality, learned Verifier training, and
dataset-level metrics/ablations. These require the corresponding GPU environments and
weights.

