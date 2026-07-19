# Change-Agent

Change-Agent is a GT-free iterative change-detection research framework built around
the existing `SegAgent` and `OmniOVCD` repositories. The runtime loop keeps T1/T2
semantic masks, instances, matching, and model evidence inside the Environment. The
Agent receives T1, T2, the predicted T1/T2 object masks, the current change mask,
structured Verifier feedback, and a compact history. These masks are model predictions,
not GT.

The implementation follows [`../CHANGE_AGENT_DEVELOPMENT_SPEC.md`](../CHANGE_AGENT_DEVELOPMENT_SPEC.md).

本轮完整开发记录见：[DEVELOPMENT_WORK_REPORT.md](DEVELOPMENT_WORK_REPORT.md)。

## Current milestone

The current code implements the v0–v3 research skeleton:

- strict `State`, `Action`, and `VerifierOutput` protocols;
- JSON extraction plus view/action/coordinate/box validation;
- a Qwen3-VL-2B Adapter using `AutoProcessor`, `apply_chat_template`, and
  `Qwen3VLForConditionalGeneration`;
- a single public `[0,1000]` coordinate protocol for Agent/Verifier and pixel XY
  coordinates only inside the Environment;
- an inference-only Environment with no GT argument or label state;
- SimpleClick point and SAM3 box boundaries;
- per-step instance extraction, default OmniOVCD overlap-presence matching, optional
  one-to-one greedy ablation, and change-mask reconstruction;
- a Qwen3-VL zero-shot Verifier that shares the Agent model weights, plus a transparent
  rule Verifier ablation and a trainable frozen-feature Verifier head for quality,
  error-map, and error-type only;
- offline GT perturbations for Verifier supervision;
- feedback-driven iteration, finish rejection, complete trajectory artifacts, and
  history-best state selection.

The three-sample LEVIR runner in `tools/run_levir_change_agent.py` now performs fresh
SAM3 text-prompt initialization for both temporal views on every sample; cached masks
are not accepted by the entry point. It saves the initial masks, confidence maps,
presence/object scores, prompt/configuration, stdout/stderr, and worker report before
running Qwen3-VL Agent actions. Qwen3-VL zero-shot verification is the default and the
rule Verifier remains an explicit `--verifier rule` ablation.

The offline training schema deliberately has no `target_view` target. Earlier smoke
data alternated T1/T2 by sample index, which was not a real label and must not be used
for training. A future trained target-view policy requires real supervision or a
separately validated latent/tool-ranking objective.

## Coordinate boundary

- Agent JSON coordinates and Verifier `error_region` are normalized XY/XYXY integers
  in `[0,1000]`. The runtime owns this protocol; Agent actions contain only the action
  fields and coordinates and do not repeat coordinate configuration metadata.
- `coordinate_frame=normalized_1000_xy` remains accepted only for compatibility with
  older action payloads. It is no longer required or requested from Qwen; a conflicting
  legacy value is rejected rather than allowed to override the system protocol.
- `ActionParser` is the only public-to-internal conversion boundary.
- Parsed actions, Environment state, and SimpleClick use original-image pixel XY.
- Trajectories preserve the raw normalized payload, parsed pixel action, and a warning
  after two consecutive actions whose public values are all `<=255`; no automatic
  pixel-coordinate correction is performed.
- SAM3 geometric prompts use normalized center/size values created by the Executor.

## Verifier feedback boundary

- Qwen's first Verifier response contains only absolute `quality_score`, independent
  pairwise `progress_score`, error type, and concise feedback. `progress_score` is in
  `[-1,1]`: positive means the candidate improved over the previous valid state,
  negative means it regressed, and zero means no material change. Initial verification
  has no comparison state and must return zero progress.
- T1/T2 originals are fixed inputs. Candidate verification additionally receives the
  previous valid T1/T2/change masks, candidate T1/T2/change masks, and the normalized
  action that produced the candidate. The final candidate change mask is the primary
  evaluation target; temporal object masks are supporting predictions rather than GT.
- The prompt explicitly defines added buildings, disappeared buildings, unchanged
  buildings, and unchanged background. Empty predicted T1 or T2 masks are not treated
  as automatic errors; the original images and final change mask decide plausibility.
- Every actionable diagnosis uses a separate localization request for `target_view`
  and `error_region`. If localization fails, the result is marked with
  `verifier_valid=false` and `localization_valid=false`, exposes no suggested action,
  and cannot stop the episode; the previous valid feedback is retained for context.
- `accept`, `stop`, and `suggested_action` are derived by the runtime. `none` maps to
  `finish` with `accept=(quality_score >= threshold)`; false positives map to a
  `negative_point`, false negatives to a `positive_point`, and mixed/uncertain errors
  to a `box` when a valid region is available.

The runner supports `verifier_best`, `conservative_best`, and `initial` selection
policies. All attempted candidate masks are retained, and initial, verifier-best,
last-attempted, and selected prediction masks are exported. A tool candidate is accepted
only when the Verifier is valid, its pairwise progress exceeds `selection_epsilon`,
and its absolute mask-area jump stays within the configured limit. Rejected candidates
remain auditable in the trajectory, while the next Agent step resumes from the previous
accepted state. If model action retries are exhausted, the episode stops without
executing a synthetic SAM3 box action.

The Agent prompt injects only the JSON example for the Verifier's current suggested
action, keeping the Qwen3-VL-2B instruction short. Point examples always include
`coordinate`, box examples always include `box`, finish examples include neither, and
none includes `coordinate_frame`. On validation failure, the retry path also supplies
the previous invalid payload and repeats an exact same-view/same-action repair template
for a missing or malformed `coordinate`/`box` field.

## Local editing and safety gates

- A positive point merges only the tool-predicted connected component containing the
  click. A negative point removes only the current-mask component containing the click.
  Unrelated components from a global SimpleClick prediction are discarded.
- A SAM3 box result replaces pixels only inside the requested pixel XYXY box; pixels
  outside that ROI are copied from the previous target-view mask.
- Every tool result records target-mask XOR statistics: action ROI, changed and
  outside-ROI pixels, outside-ROI ratio, target-mask change ratio, largest changed
  component, and before/after component counts.
- Candidate acceptance rejects excessive outside-ROI changes, excessive target-mask
  changes, and excessive component-count jumps in addition to the existing progress and
  change-mask area gates. Thresholds are explicit runner arguments and trajectory fields.
- Verifier localization rejects near-full-image regions unless the candidate delta is
  itself broad. False-positive regions must overlap current white change pixels, while
  false-negative regions must lie mostly outside them. Invalid localization is retried
  with the exact consistency error before the Verifier is marked invalid.

## Runtime audit

Each sample writes `episode_summary.json` with its stop reason, loop count,
accepted/rejected candidates, tool count, invalid action attempts, elapsed time, and
selected steps. Every invalid action attempt records loop/attempt indices, raw output,
validation error, prompt SHA-256, and timestamp. Trajectory metadata resolves Git from
the repository path and records commit plus dirty state. The run manifest records Python,
platform, parent-process seeds, model metadata checksums, and deterministic-policy state.
Isolated segmentation workers are launched through `seeded_segmentation_worker.py`,
which seeds Python, NumPy, and PyTorch before model construction and appends that seed
record to each worker report.

## Local smoke commands

The isolated `omniovcd-env` can prepare the Qwen3-VL dependency set without changing
the legacy SegAgent environment:

```bash
/Data/wyh/CD-SegAgent/omniovcd-env/bin/pip install \
  'accelerate>=0.26' 'qwen-vl-utils>=0.0.8'
HF_ENDPOINT=https://huggingface.co \
  /Data/wyh/CD-SegAgent/omniovcd-env/bin/python tools/download_qwen3vl.py
```

On a host without a working NVIDIA driver, load and generate on CPU:

```bash
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/qwen3vl_smoke.py --device-map cpu
```

The smoke report records load time, process RSS, CUDA availability, raw JSON, and the
parsed Action. On a CUDA server use `--device-map auto`; the report then includes
PyTorch allocated/reserved VRAM.

Verifier and adapter-loop smoke tests do not need model weights:

```bash
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/train_verifier.py --smoke
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/rollout_smoke.py
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/rollout_smoke.py --agent qwen3vl \
  --model-path /Data/wyh/CD-SegAgent/models/Qwen3-VL-2B-Instruct \
  --device-map cpu
```

`configs/runtime_cpu.json` documents the no-GPU fallback. Use
`tools/collect_runtime_manifest.py` at the start of real runs to capture the commit,
software, model path, split, seed, and CUDA state.

On this host, GPU access requires an execution context with `/dev/nvidia*` mounted.
The validated low-occupancy profile is one L20 only:

```bash
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/rollout_smoke.py --agent qwen3vl --device-map auto \
  --model-path /Data/wyh/CD-SegAgent/models/Qwen3-VL-2B-Instruct
```

Use [`configs/runtime_gpu_l20.json`](configs/runtime_gpu_l20.json) as the corresponding
runtime manifest. A shell launched only inside the restricted filesystem sandbox can
still report `cuda.is_available() == false`; verify with `nvidia-smi` in the device-
enabled execution context.

Real local-weight adapter checks are isolated by dependency stack:

```bash
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/sam3_smoke.py --device cpu --resolution 1008
PYTHONPATH=/Data/wyh/CD-SegAgent/change_agent:/Data/wyh/CD-SegAgent/SegAgent:/Data/wyh/CD-SegAgent/SegAgent/third_party/SimpleClick \
  /Data/wyh/CD-SegAgent/segagent-env/bin/python \
  tools/simpleclick_smoke.py --device cpu
```

Prepare real offline Verifier samples and run a bounded training smoke:

```bash
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/build_verifier_samples.py \
  --dataset-root /Data/wyh/CD-SegAgent/OmniOVCD/dataset/LEVIR-CD/test_256 \
  --max-samples 2 --output outputs/verifier_levir_smoke.npz
PYTHONPATH=. /Data/wyh/CD-SegAgent/omniovcd-env/bin/python \
  tools/train_verifier.py --samples outputs/verifier_levir_smoke.npz \
  --epochs 1 --batch-size 1 --output outputs/verifier_levir_smoke.pt
```

## Environment isolation

Do not upgrade the legacy SegAgent environment (`transformers==4.31.0`). Install the
new Agent branch in a separate environment:

```bash
python -m pip install -e '.[qwen3vl]'
```

Install training dependencies separately when training the Verifier:

```bash
python -m pip install -e '.[train]'
```

OmniOVCD/SAM3 and SimpleClick should remain in their existing isolated environments.
Their adapters accept injected callbacks/wrappers so incompatible CUDA and
Transformers stacks do not need to share one process.

## Run tests and smoke loop

The test suite only requires NumPy and Pillow:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python tools/smoke_loop.py --output outputs/smoke
```

The smoke output includes `trajectory.json` and one `.npy` change mask per step. Run
metadata records the Git commit, Python/platform, seed, and dataset split. Production
runs should additionally pass config/checkpoint/GPU/software metadata through
`ChangeAgentEnvironment(run_metadata=...)`.

## Adapter contract

`OmniOVCDAdapter` receives two injected callbacks:

1. `initialize_masks(t1_image, t2_image, query)` returns T1/T2 masks plus optional
   no-GT model evidence.
2. `segment_box_callback(image, normalized_cxcywh, query)` executes SAM3 geometric
   prompting.

`SAM3ProcessorAdapter` provides those callbacks directly from OmniOVCD's public
`SAM3ImageProcessor` API. It uses text prompts to initialize both temporal masks and
`add_geometric_prompt` for normalized box actions, while keeping processor state and
features hidden from the Agent.

`SimpleClickAdapter` calls the underlying SimpleClick predictor directly so the point
session's initial target-view mask is actually supplied as `prev_mask`. Because real
tools run in isolated subprocesses, the Environment stores a separate T1/T2 session
base mask and accepted click history; each worker reconstructs the Clicker and replays
those clicks in order before applying the candidate click. Rejected candidates are not
committed, and an accepted box starts a new point session from the box-edited mask.
Every tool result is checked for shape and locality before it can update live state.

The upstream OmniOVCD checkout contains three CPU-compatibility fixes used by the
adapter smoke: device-aware SAM3 decoder coordinate caches, CPU-safe positional
encoding precomputation, and a non-pinned CPU geometry scale path. These are committed
in that checkout only; model weights remain outside Git.

## Safety boundary

Runtime `ChangeAgentEnvironment` only accepts `inference_only=True`; `reset` has no GT
parameter. GT-derived perturbations and labels live in `perturbations.py`, which is an
offline training utility and is never imported by the Environment.
