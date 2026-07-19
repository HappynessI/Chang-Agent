# Development log

## 2026-07-19 — compact delta-effect verifier and duplicate-candidate safety

The `change_agent_levir_gpu_closed_loop_20260719_173613` audit exposed three
structural failures: a white `candidate_added` component could be called a false
negative, identical candidates received inconsistent pairwise labels, and six verbose
region feedback sentences could exceed the Qwen3-VL-2B output budget.

- Initial region output is now compact: each fixed `rN` maps to
  `[verdict,target_view]`. Per-region prose is no longer requested. An ordinary white
  component cannot be labeled `false_negative`; that label requires a
  `temporal_difference_missing` proposal.
- Candidate verification no longer repeats the initial six-region analysis or asks Qwen
  to emit `better/worse`. The Environment supplies at most two actual delta panels—one
  aggregating additions and one aggregating removals—preserves polarity, and asks only for `added_supported`,
  `added_unsupported`, `removed_supported`, `removed_unsupported`, or `uncertain`.
- The runtime derives comparison deterministically. Any harmful effect makes the
  candidate worse, any uncertainty remains uncertain, and every inspected effect must
  be beneficial before the candidate can be better. A delta with uninspected pixels is
  invalid and therefore rejected.
- Candidate decisions are cached by a SHA256 over previous masks, candidate masks, and
  action. The Environment separately fingerprints actions and refuses an exact action
  that was already rejected on the same live state.
- Saved-candidate replay now reconstructs local point/box composition rather than using
  a worker's raw full-image mask as the temporal candidate mask.
- Verifier output budget defaults to 1024 tokens. The compact schema remains mandatory;
  the larger ceiling provides headroom for occasional Qwen formatting drift and retries.
  Trajectory and run-manifest source
  metadata now includes `git_worktree_sha256`, covering tracked diffs and untracked file
  contents without persisting the diff itself.

CPU regressions cover compact initial labels, impossible white false negatives,
programmatic beneficial/harmful comparisons, candidate cache hits, delta polarity and
coverage, repeated-action rejection, and worktree fingerprinting. No GPU rollout was
started as part of this implementation; current behavior still requires a saved-candidate
replay followed by the fixed three-sample closed loop.

## 2026-07-19 — replay challenge, safe initial finish, and pairwise delta crops

- Added `tools/replay_verifier_challenge.py`, which replays the saved tool candidates
  from the previous three-sample run through the GT-free Verifier. GT is opened only
  after each Verifier response to assign offline `better/worse/unchanged` labels.
  The report is atomically committed, so a failed model/configuration run removes its
  temporary directory and leaves no failed result under `outputs/`.
- The Environment and Agent now allow a direct initial `finish` only when the initial
  Verifier is valid, reports `error_type=none`, and sets `comparison=initial, stop=true`.
  Invalid or actionable initial feedback still requires a segmentation tool action.
- Pairwise local panels now show previous-vs-candidate change pixels explicitly:
  previous red, candidate green, and delta blue, alongside the T1/T2 and semantic-mask
  views. This makes the actual candidate edit visible at the same crop scale used for
  regional diagnosis.
- Added regressions for safe initial finish, initial Agent prompt authorization, local
  delta visibility, atomic failure cleanup, and declared comparison tolerances.

## 2026-07-19 — region-grounded diagnosis and categorical pairwise gate

- Replaced Qwen's joint absolute `quality_score` plus continuous `progress_score`
  prediction with two focused stages. The regional stage classifies fixed proposals;
  the candidate stage emits only `better`, `worse`, `unchanged`, or `uncertain`.
  Qwen runtime outputs now leave both numeric score fields null.
- Added Environment-owned Verifier proposals from change-mask connected components,
  T1/T2 object-mask XOR components, and candidate added/removed delta components.
  Proposals are padded, deduplicated, capped at six, serialized with their sources and
  exact pixel counts, and converted to normalized boxes by the runtime.
- Preserved the five labeled full-image inputs and added a 384x384 local four-panel view
  per proposal. The panel explicitly enlarges the RGB crops, binary change mask, and
  color-coded temporal masks, so small white components remain visible.
- Region responses must cover every Environment `region_id` exactly once with
  `true_change`, `false_positive`, `false_negative`, or `uncertain`. Qwen no longer
  generates localization coordinates; actionable `error_region` is the selected
  Environment proposal box.
- Added authoritative mask facts and a semantic consistency gate: when the runtime
  counts any white change pixels, a model response claiming the mask is empty is invalid
  and retried. The proposal builder retains even a single white pixel despite the normal
  minimum-area filter.
- Environment candidate commit now requires categorical `comparison=better` while
  retaining verifier-validity, locality, area, topology, rollback, and full trajectory
  gates. Pairwise trajectories use the latest accepted state as best instead of
  manufacturing an absolute scalar rank.
- Added regressions for proposal construction/source merging, one-pixel preservation,
  full-plus-local visual inputs, empty-mask contradiction retries, fixed-box diagnosis,
  pairwise-only output, identical-state validation, and rejection of worse candidates.

Validation:

```text
Python byte compilation: passed
Unit tests: 68 passed
git diff --check: passed
```

## 2026-07-19 — restore the official SAM3 CUDA autocast contract

- The first migrated A800 smoke job (`40907`) reached fresh SAM3 initialization but
  failed in the fused ViT MLP with a BF16-activation/FP32-weight dtype mismatch.
- The isolated segmentation worker had omitted the CUDA autocast context used by the
  official SAM3 inference examples. SAM3 text initialization and box inference now
  run inside a scoped autocast context: BF16 on supported GPUs and FP16 otherwise.
  CPU execution keeps a no-op context.
- Added regression coverage for CPU, CUDA BF16, and CUDA FP16-fallback selection.
  Slurm released the failed job's GPU; its diagnostic output was removed after the
  final successful rollout, as requested during output cleanup.
- The follow-up job (`40909`) passed the fused MLP and exposed a separate normal
  zero-detection edge case: SAM3 can return empty semantic, instance, and mask arrays.
  The adapter now maps an explicit empty detector result to an all-zero mask and
  confidence map while still raising when the state contains no mask output keys.
  Added a regression for this distinction. Slurm released the failed job's GPU; its
  diagnostic output was removed after the final successful rollout.
- Job `40910` then completed inference and failed only while persisting a BF16
  diagnostic tensor, because PyTorch cannot directly expose BF16 storage to NumPy.
  The adapter now promotes BF16 to FP32 strictly at the diagnostic serialization
  boundary, with a real BF16 tensor regression. Its GPU was released, and the failed
  diagnostic output was removed after the final successful rollout.
- The final one-GPU job (`40911`) completed all three fixed LEVIR-CD samples on one
  A800 in 114 seconds and released its allocation. Aggregate conservative-selected
  IoU/F1 were `0.69744116`/`0.82175592`. The run confirmed that field-free point
  actions execute, retry exhaustion no longer invokes a synthetic box, and rejected
  candidates roll back without contaminating the selected prediction.
- Remaining audit findings are model-policy issues rather than runtime failures:
  ten Agent responses omitted required point/box coordinates, all six Verifier
  localizations covered the whole image, and a post-rollout-improving candidate for
  `test_85_16` (IoU `0.30658070` to `0.38875878`) was rejected because the GT-free
  Verifier score stayed at zero. See
  `outputs/change_agent_levir_gpu_smoke_20260719_030624/CLOSED_LOOP_AUDIT.md`.

## 2026-07-18 — expose both temporal object masks to the Verifier

- Expanded every Qwen Verifier request from three to five labeled visual inputs:
  `T1 original image`, `T2 original image`, `Predicted T1 object mask`,
  `Predicted T2 object mask`, and `Current change mask`.
- The diagnostic prompt explicitly states that T1/T2 masks are predictions rather than
  GT and that the change mask is reconstructed from those masks and OmniOVCD matching.
- Added a regression asserting the exact input order and labels, so future prompt changes
  cannot silently make the model infer image/mask roles from position alone.

## 2026-07-18 — no synthetic box fallback and rejected-candidate rollback

- Removed the runner-generated SAM3 box action that previously executed after Qwen
  action retries were exhausted. Exhaustion now records all rejected raw outputs,
  sets `episode_stop_reason=action_retry_exhaustion_without_state_change`, and safely
  exports the current/history-best state even when no tool action ran.
- Added an Environment candidate-commit boundary. Tool candidates are rejected when
  the Verifier is invalid, the score does not improve beyond `selection_epsilon`, or
  the absolute change-mask area jump exceeds `max_selection_area_delta`.
- Rejected candidate masks, Verifier outputs, tool evidence, area ratios, and rejection
  reasons remain in the trajectory, but the live Environment state and valid feedback
  roll back to the previous accepted version. The step index still advances so repeated
  failures cannot create an infinite loop.
- History-best selection excludes rejected candidates; raw `verifier_best` remains an
  audit artifact. Added regressions for retry exhaustion without fallback execution,
  invalid-Verifier rollback, and excessive-area rollback.

## 2026-07-18 — staged Verifier diagnostics and derived actions

- Split Qwen Verifier handling into a diagnostic stage and a dedicated localization
  request. Missing `error_region` is retained as a semantic diagnosis instead of
  being discarded immediately; one localization request is issued before declaring
  the Verifier invalid.
- Removed model authority over `accept`, `stop`, and `suggested_action`. The runtime
  derives them from `error_type`, `error_region`, and the quality threshold. Invalid
  feedback is explicitly marked `verifier_valid=false`, has
  `localization_valid=false`, `suggested_action=null`, and `stop=false`.
- Invalid feedback retains the previous valid diagnostic text and never tells the
  Agent to finish. Environment termination now uses the derived `stop` field.
- Added regressions for contradictory model fields, successful second-stage
  localization, failed localization with retained feedback, and invalid no-finish
  fallback behavior.

## 2026-07-18 — system-owned Agent coordinate protocol

- Removed `coordinate_frame` from the Qwen action schema and the runner's safety
  fallback. The prompt still defines normalized `[0,1000]` XY/XYXY, while the model
  now emits only action fields and coordinates.
- Made the retired field optional at the parser boundary. Correct legacy declarations
  remain accepted for old artifacts/clients, conflicting values remain invalid, and
  missing declarations no longer consume retries or trigger safety fallbacks.
- Added point, box, prompt, Environment, and trajectory regressions for field-free
  actions. This directly covers the five valid point actions rejected in
  `change_agent_levir_fresh_qwen_20260718_133335` solely for omitting the field.

## 2026-07-18 — normalized protocol, fresh SAM3, and zero-shot Verifier

- Unified every Agent/Verifier-facing region under normalized `[0,1000]` XY/XYXY;
  pixel coordinates now exist only after `ActionParser` inside the Environment.
- Required `coordinate_frame=normalized_1000_xy` for every point/box Agent action;
  missing frames are rejected, repeated all-`<=255` values produce an audit warning,
  and no automatic pixel correction is attempted.
- Added configurable `verifier_best`, `conservative_best`, and `initial` selection;
  every run exports initial, verifier-best, last, and selected masks.
- Added explicit coordinate-space metadata and regression coverage for the 256-pixel
  coordinate mismatch observed in the first three-sample rollout.
- Removed the cached-mask option from the LEVIR runner. Each sample now starts a fresh
  dual-view SAM3 text-prompt worker that loads SAM3 once, processes T1/T2, and saves
  both masks plus all confidence/presence/object-score arrays and worker parameters.
- Added a structured, GT-free Qwen3-VL zero-shot Verifier that shares the already
  loaded Agent model/processor, preserves raw verifier responses, and infers
  `target_view` visually instead of alternating labels.
- Removed the synthetic `index % 2` target-view labels, target-view training array,
  classifier head, action head, and their losses. Training schema/checkpoints are now
  version 3 and retain only quality/error-map/error-type supervision.
- SAM3 initialization persistence is limited to selected diagnostics (mask/logit,
  confidence/presence/object scores, and selected FPN feature arrays), not every
  Transformer activation or attention map.
- Fixed fresh SAM3 mask construction after the first live attempt exposed that
  `semantic_mask_logits` are already probabilities. The adapter now mirrors
  OmniOVCD's semantic/instance/object/presence fusion and `prob_thd=0.4` threshold,
  instead of treating every positive probability as foreground.
- Enforced the first-tool requirement at the Environment boundary after a live retry
  showed that prompt-only guidance allowed three consecutive `finish` actions. Agent
  retries now include the exact validation error, including missing coordinate frames.
- Added an explicitly logged, mask-bounded SAM3 box safety fallback after all action
  retries are rejected, so a zero-shot episode can finish without treating `finish` as
  an executable segmentation decision.
- Made the Qwen zero-shot Verifier return an explicit `uncertain_region` safe fallback
  after malformed structured outputs exhaust retries; raw output, validation errors,
  and `fallback=true` remain in verifier evidence instead of aborting the rollout.
- Kept `RuleBasedVerifier` only as the explicit `--verifier rule` ablation; the real
  runner defaults to `qwen_zero_shot`.
- Renamed offline report fields to `verifier_selected_step` and
  `verifier_selected`, avoiding an implication that the selection is GT-oracle best.

Validation:

```text
Python byte compilation: passed
Unit tests: 33 passed
Runner/worker CLI parsing: passed
git diff --check: passed
```

## 2026-07-17 — overlap-presence matching and full LEVIR runner

- Changed the default matching mode from one-to-one greedy to OmniOVCD-compatible,
  directional `overlap_presence` with threshold `0.25`.
- Kept `greedy_one_to_one` as an explicit ablation and added candidate-pair,
  directional-coverage, and split/merge-ambiguity evidence.
- Separated `t12_min_instance_area` and `cd_min_instance_area`; both default to zero.
- Added isolated real-model subprocess adapters and a segmentation worker for
  SimpleClick point actions and SAM3 box actions.
- Added `tools/run_levir_change_agent.py` for the three fixed LEVIR-CD samples. It
  saves per-step masks/trajectories, verifier feedback, tool reports, history-best
  predictions, and performs GT evaluation only after a rollout-complete marker.
- Expanded the matching regression suite to cover split/merge, additions,
  disappearances, unrelated instances, directional coverage, and area filters.

Validation before the full GPU run:

```text
Python byte compilation: passed
Unit tests: 19 passed
git diff --check: passed
```

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

The model-level checks below are now validated with local weights. They are kept in
separate processes because SAM3 and SimpleClick use incompatible dependency stacks.

## 2026-07-17 — training/rollout/server smoke preparation

- Added a deterministic `train_verifier.py --smoke` path. It exercises the trainable
  head, quality/error-map/classification losses, optimizer update, checkpoint writing,
  and memory telemetry.
- Added `rollout_smoke.py`, which uses the real `OmniOVCDAdapter` boundary and mock
  tool callbacks to exercise two feedback-loop steps without GT.
- Added `SAM3ProcessorAdapter` for concrete `set_image`, `set_text_prompt`, and
  `add_geometric_prompt` integration, with a fake-processor regression test.
- Added `qwen3vl_smoke.py` and `download_qwen3vl.py`; the latter keeps model files
  under `models/`, outside the Git repository.
- Added `collect_runtime_manifest.py` and `configs/runtime_cpu.json` for reproducible
  local/server runs.

Smoke results on the current host:

```text
Verifier: 2 epochs, loss 5.3406 -> 5.3378, RSS ~630 MB, CUDA unavailable
Adapter rollout: 2 steps, RSS ~38 MB, CUDA unavailable
```

The current host reports no NVIDIA driver (`nvidia-smi` cannot communicate with the
driver), so real GPU VRAM and CUDA model generation remain pending. Qwen3-VL download
is being performed from the official Hugging Face endpoint in tmux; its completion
status and artifact path are recorded in `logs/qwen3vl_download_official.log`.

The local Qwen3-VL checkpoint subsequently downloaded successfully to
`/Data/wyh/CD-SegAgent/models/Qwen3-VL-2B-Instruct` (4,255,140,312-byte
`model.safetensors`). The real CPU generation smoke passed:

```text
load_seconds: 5.115
RSS: 37.87 MB -> 981.57 MB (delta 943.70 MB)
raw JSON: {"target_view":"t1","action":"positive_point","coordinate":[500,100]}
parsed pixel coordinate: [32, 6]
transformers: 4.57.3
```

The smoke process reported a transient CUDA probe inconsistency (one process saw 8
devices, while an independent probe and `nvidia-smi` could not initialize NVML). Since
the run used `device_map=cpu` and allocated/reserved 0 MB VRAM, no GPU memory claim is
made until the server driver is healthy.

The Qwen3-VL-to-Environment rollout smoke also passed. The generated raw response was
accepted by the ActionParser, executed as an Environment step, and followed by a
Verifier-accepted `finish`; the trajectory was saved under
`/tmp/change_agent_qwen_rollout_smoke2/trajectory.json`. A single-GPU probe with
`CUDA_VISIBLE_DEVICES=0` currently reports no usable CUDA device, so no GPU was
allocated. When the driver is repaired, use one visible GPU and `device_map=auto` for
the same script; do not expose all GPUs for this smoke.

Real-weight adapter and Verifier checks:

```text
SAM3 text + geometric prompt (CPU): 50.949 s, peak RSS ~7,493 MB,
  models/sam3/sam3.pt, text masks 0, box masks 1
SimpleClick ViT-L external-mask point (CPU): 10.807 s, peak RSS ~3,124 MB,
  models/SimpleClick/cocolvis_vit_large.pth, output 256x256
LEVIR-CD verifier sample builder: 2 paired samples, 10 feature channels
LEVIR-CD verifier head: 1 epoch, loss 5.6379 -> 5.6904, peak RSS ~864 MB
```

The SAM3 and SimpleClick commands intentionally run independently; loading both
large checkpoints together is outside the bounded smoke-test memory budget. The
Qwen-to-Environment smoke remains the supported full Agent rollout, while the two
segmentation adapters are validated at their real model boundaries.

GPU validation (one visible L20, `CUDA_VISIBLE_DEVICES=7`):

```text
Qwen3-VL smoke: CUDA allocated ~4,067 MB, peak ~4,147 MB, load 6.893 s
Qwen3-VL → Environment rollout: CUDA allocated ~4,067 MB, peak ~4,147 MB,
  1 step, 14.188 s
SAM3 text + box smoke: CUDA allocated ~3,525 MB, peak ~3,909 MB, 13.540 s
SimpleClick point smoke: CUDA allocated ~1,249 MB, reserved ~1,450 MB,
  peak ~1,390 MB, 7.167 s
```

The earlier false “CUDA unavailable” result came from running Python inside the
restricted sandbox, which does not expose `/dev/nvidia*`; `nvidia-smi` and the same
commands in the device-enabled context see eight NVIDIA L20 GPUs with driver
595.58.03.
