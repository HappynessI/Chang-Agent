# Staged Verifier protocol maintenance

## Scope

The staged verifier is an opt-in alternative to the legacy rich single-pass
Qwen verifier.  It separates evidence inspection, diagnosis, action planning,
and candidate comparison while keeping all proposal geometry and editability
facts under Environment control.

The diagnosis prompt must remain model-capability neutral.  It must not encode
the old small-model shortcut that a white change region with a visible T1/T2
difference is normally correct.  Current diagnosis asks the MLLM to inspect
whole-region mask coverage, boundaries, and internal gaps; `mixed_error` is
valid when one proposal contains both supported and unsupported pixels.  The
runtime keeps geometry, schema, editability, and polarity checks, but does not
replace MLLM semantic judgment with a default `none` decision.

The protocol implementation is split across:

- `change_agent/verifier_protocol.py`: typed records and enums;
- `change_agent/adapters/staged_verifier.py`: state machine and cross-stage checks;
- `change_agent/adapters/stage_backends.py`: local Transformers and BaiLian
  OpenAI-compatible stage backends;
- `change_agent/adapters/bailian_adapter.py`: hosted Agent action generation;
- `tools/run_levir_change_agent.py`: CLI selection and run-manifest metadata.

## Stage contract

The initial-state path is:

1. Environment creates stable proposals, normalized geometry, mask occupancy,
   and editable-seed facts.
2. `evidence` classifies only T1/T2 visual state and evidence quality.
3. `diagnosis` emits one supported error type and target view.
4. `plan` selects an executable action using supplied geometry.
5. `decision` emits the initial quality/accept/stop judgment.

The candidate path uses `candidate_evidence`, `candidate_diagnosis`, and
`decision` with both previous and candidate change masks.  Candidate `accept`
may be true only when `comparison=better`; `better` alone must not override a
model's explicit rejection. An identical candidate is
handled programmatically as `unchanged` without a model call.

The Environment commits a non-finish candidate only when all runtime gates
pass, `comparison=better`, and the verifier also returns `accept=true`.
`better, accept=false` remains a recorded rejected candidate and the next turn
continues from the prior accepted masks.

Rollback preserves only the prior accepted masks and accepted point-session
history. It must not reuse a rejected action instruction. Proposal and Hybrid
regenerate their Agent action from the retained trajectory history. Direct calls
its full-context verifier in `replan` mode, with the rejected action, candidate
mask delta, candidate verdict, rejection reasons, and a bounded four-entry
rejection history. The current images/masks in that call are the accepted state;
additional masks are explicitly labelled as the rejected candidate. Its output
has `comparison="uncertain"` and `accept=false`, and must author a different
action or `finish`. If this replan is invalid, no action is authorized.

Each model call has a minimal exact JSON schema.  The runtime rejects unknown
fields, unknown region IDs, non-enum values, string booleans, non-integer public
coordinates, points outside the proposal, negative clicks on black object-mask
seeds, and positive clicks on white seeds.  Invalid output never authorizes a
tool action or finish.

Runtime has no hard semantic mapping from a coarse T1/T2 state pair and a
white/black Proposal to `none`, `false_positive_change`, or
`false_negative`. A white component can contain both a real appearance change
and unsupported boundary/interior pixels. The runtime validates only structural
polarity (`false_positive_change` needs a white proposal and
`false_negative` needs a black proposal), then retains Qwen's semantic
`false_positive_change` or `mixed_error` result for planning.

`diagnosis` and `candidate_diagnosis` receive actual visual inputs, not only a
serialized prior evidence judgment. In `hybrid` mode they receive full T1/T2,
object masks, change mask, and exact regional crops. In `proposal` mode they
receive the exact regional RGB/object-mask/change-mask crops only.

## Backends

`LocalQwen3VLStageBackend` reuses already-loaded local Qwen model/processor
objects.  `BailianQwen3VLStageBackend` uses the OpenAI-compatible chat endpoint,
`response_format={"type":"json_object"}`, and base64 PNG inputs.  Both satisfy
the same `StageBackend.generate_stage` interface and return provider-independent
Python mappings.

Local Transformers decoding does not provide a provider-side JSON constraint.
The staged backend therefore uses stage-aware extraction: it scans all complete
JSON objects and selects the envelope required by the current stage (`evidence`,
`diagnosis`, `plan`, or `decision`). It never accepts the first object merely
because it is valid JSON. The prompt places the output contract before a
delimited Environment-facts envelope, and a schema/semantic validation failure
is sent back to the same stage for a bounded repair attempt. Repair does not
relax the typed protocol; the response must still pass the exact stage parser.
Each staged call records a bounded raw response, parsed output, prompt hash,
latency, and validation error in `backend_calls` for debugging. Credentials
are not included in these records.

The local 2B GPU smoke run showed that diagnosis generation deterministically
omitted `confidence` on both the initial and repair attempt while preserving a
valid `error_type` and `target_view`. Diagnosis confidence is therefore an
optional ranking hint: an omitted value is normalized conservatively to `0.0`.
The action-bearing fields remain required and strictly validated; this
normalization cannot authorize an invalid target view, action, or geometry.

The next local GPU smoke run reached `plan` and exposed prompt-induced action
bias: a static `positive_point` example was copied even though the selected T2
seed was white. Plan templates are now derived from authoritative Environment
facts. A white seed produces a `negative_point` contract, a black seed produces
a `positive_point` contract, and the contract contains the exact component
seed. Runtime editability and geometry validation remain mandatory.

The following GPU smoke run reached candidate decision, where a static
`comparison=initial` example caused the local model to repeat an invalid label.
Decision templates are now mode-aware: initial uses `initial`; candidate uses
the valid non-initial `uncertain` example, while the parser still requires
candidate `accept=true` to require `comparison=better`, without converting an
explicit `accept=false` into acceptance.

The BaiLian path did not exhibit the copied-context parsing failure in the
smoke run because its request uses server-side `response_format=json_object`,
which returns one JSON message content. This constrains JSON syntax, not the
full application schema: the BaiLian run still exposed a semantic
`target_view` validation error and a later HTTP 400 candidate request, so it
uses the same stage-aware parser and repair interface.

Hosted vision providers also impose a minimum image side length.  Region
proposals are derived from connected components and can be smaller than that
limit even when the source LEVIR-CD images are 256x256.  Before a staged
BaiLian request, `_normalized_crop_box` expands only the provider-facing PIL
crop to at least 11x11 pixels, clamped to the source image.  Proposal geometry,
mask facts, and verifier semantics remain unchanged.

The hosted backend reads credentials only from an environment variable.  The
default is `DASHSCOPE_API_KEY`; the key is never included in trajectory metadata,
errors, or `last_call`.  Configure the endpoint through `--bailian-base-url` or
`DASHSCOPE_BASE_URL`.  A workspace-specific BaiLian base URL is preferred when
available.  Do not place API keys in command lines, JSON configs, CSV files, or
the repository.

`tools/run_with_bailian_csv.py` accepts the exported BaiLian workspace CSV only
as a local secret source: it loads the `apiKey` row into the child process and
sets `DASHSCOPE_BASE_URL` from the `openAiCompatible` row.  This is required for
workspace-scoped keys; the public DashScope endpoint is not a substitute.
Provider HTTP errors retain only a short, credential-redacted diagnostic.

For Slurm runs, probe the candidate node's workspace, model/tool paths, GPU,
proxy fingerprints, and workspace endpoint first, then submit with the same
node pinned via `--nodelist`.  `tools/submit_ca0720_bailian_fix.sh` requires the
probed node in `NODE` and requires `BAILIAN_NETWORK_MODE=direct|proxy`.  Use
`direct` only when the probe succeeds after unsetting proxy variables; use
`proxy` only when the same node has a working proxy listener and matching
fingerprints.  The helper archives Slurm stdout/stderr under the experiment
output's `logs/` directory before cleaning temporary `/tmp` copies.

## Runner modes

The runner exposes independent Agent and Verifier backend selection:

```text
--agent-backend local|bailian
--verifier qwen_zero_shot|qwen_staged|rule
--staged-verifier-backend local|bailian
--bailian-model qwen3-vl-plus
--proposal-mode direct|proposal|hybrid
```

This supports the intended 2x2 comparison:

| Agent/Verifier model | Legacy verifier | Staged verifier |
|---|---|---|
| local Qwen3-VL-2B | `local + qwen_zero_shot` | `local + qwen_staged/local` |
| BaiLian Qwen3-VL-Plus | legacy is not hosted | `bailian + qwen_staged/bailian` |

A mixed test is also supported, such as local Agent actions with a BaiLian
staged verifier. `direct` is the designated Proposal ablation: Qwen sees only
complete state and authors its action geometry; Proposals are not attached.
`proposal` uses regional crops and Environment-owned Proposal geometry. `hybrid`
uses complete state plus regional crops for diagnosis while retaining
Environment-owned Proposal geometry for execution and candidate verification.
All modes retain coordinate parsing, tool safety gates, rollback, and
offline-only GT evaluation.

## Required evaluation

Do not compare only aggregate IoU/F1.  For each model/protocol cell, report:

- stage schema-valid rate;
- diagnosis-valid and target-view-valid rates;
- executable-action rate;
- verifier-invalid and action-retry rates;
- number of tool actions and accepted/rejected candidates;
- initial versus selected per-sample IoU/precision/recall/F1;
- hosted request latency, token usage, and request ID when available.

Run the same fixed inputs, initial SAM3 artifacts, seed, proposal configuration,
and selection gates in every cell.  GT remains unavailable until rollout has
completed.

## Tests

For the current `wangyihan` deployment, runtime verification is performed only
as a short compute-node GPU smoke run. Do not launch additional CPU test runs.
Slurm stdout/stderr are temporary `/tmp` files and are removed by the job
wrapper; `outputs/` contains only experiment artifacts and structured protocol
diagnostics.

The protocol tests are CPU-only and do not call external services:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  /guisongxia01/pangchao/wangyihan/omniovcd-env/bin/python \
  -m unittest tests.test_staged_verifier tests.test_bailian_adapter -v
```

`test_bailian_adapter` uses an injected fake HTTP opener and a temporary fake
environment key.  It verifies JSON mode, endpoint construction, usage/request
metadata, missing-key failure, and absence of credentials from audit records.
