# Change-Agent

Change-Agent is a GT-free iterative change-detection research framework built around
the existing `SegAgent` and `OmniOVCD` repositories. The runtime loop keeps T1/T2
semantic masks, instances, matching, and model evidence inside the Environment. The
Agent receives only T1, T2, the query, the current change mask, structured Verifier
feedback, and a compact history.

The implementation follows [`../CHANGE_AGENT_DEVELOPMENT_SPEC.md`](../CHANGE_AGENT_DEVELOPMENT_SPEC.md).

## Current milestone

The current code implements the v0–v3 research skeleton:

- strict `State`, `Action`, and `VerifierOutput` protocols;
- JSON extraction plus view/action/coordinate/box validation;
- a Qwen3-VL-2B Adapter using `AutoProcessor`, `apply_chat_template`, and
  `Qwen3VLForConditionalGeneration`;
- an inference-only Environment with no GT argument or label state;
- SimpleClick point and SAM3 box boundaries;
- per-step instance extraction, one-to-one matching, and change-mask reconstruction;
- a transparent rule Verifier baseline and a trainable frozen-feature Verifier head;
- offline GT perturbations for Verifier supervision;
- feedback-driven iteration, finish rejection, complete trajectory artifacts, and
  history-best state selection.

Real OmniOVCD, SAM3, SimpleClick, and Qwen3-VL weight-level GPU integration remains a
server validation task. The rule Verifier is a baseline, not the research-result
trained Verifier.

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

`SimpleClickAdapter` wraps SegAgent's segmentation model and passes the current target
view mask as the SimpleClick `prev_mask`. Every tool result is checked for shape before
it can update Environment state.

The upstream OmniOVCD checkout contains three CPU-compatibility fixes used by the
adapter smoke: device-aware SAM3 decoder coordinate caches, CPU-safe positional
encoding precomputation, and a non-pinned CPU geometry scale path. These are committed
in that checkout only; model weights remain outside Git.

## Safety boundary

Runtime `ChangeAgentEnvironment` only accepts `inference_only=True`; `reset` has no GT
parameter. GT-derived perturbations and labels live in `perturbations.py`, which is an
offline training utility and is never imported by the Environment.
