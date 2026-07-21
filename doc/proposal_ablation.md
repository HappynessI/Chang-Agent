# Proposal ablation

Run the three BaiLian Qwen3-VL-Plus arms with the same samples, initialization,
seed, tools, matching, safety gates, and offline evaluator. GT remains closed
until every rollout finishes.

- `direct`: current BaiLian operational mode. Full T1/T2/object-mask/change-mask
  context; Qwen emits binary rubric judgments, short evidence, diagnosis, action,
  and normalized geometry. No Proposal is attached to Environment state.
- `proposal`: per-region RGB, T1/T2 object-mask, and change-mask crops only.
  Environment Proposal seed/box grounds every executable action.
- `hybrid`: full context plus same regional crops. Environment Proposal seed/box
  still grounds execution and delta verification.

`tools/submit_ca0721_proposal_ablation.sh` creates one parent directory outside
`outputs/`:

```text
experiments/CA_0721(<run>)-bailian-proposal-ablation/
  direct/
  proposal/
  hybrid/
```

Each child contains its own `logs/`, trajectories, feedback, masks,
predictions, and `per_sample_metrics.json`. Compare initial-error localization,
small-change recall, invalid/unsafe tool actions, accepted-candidate IoU/F1,
audited-region coverage, token usage, endpoint failures, and latency. Do not
rank arms from aggregate IoU alone.

Direct schema `direct_change_rubric_v3` canonicalizes common model aliases such
as `missing_detection` to `false_negative`, while retaining JSON and geometry
validation. Qwen cannot author quality, progress, comparison, or acceptance.
Runtime weights change precision/recall/extent/boundary/artifact booleans, applies
the evidence-sufficiency hard gate, retains target-scope as an audit diagnostic,
then derives candidate comparison from one
benefit and three harm flags. Proposal and Hybrid retain their staged decision
contract. In every arm, Environment commits only `better, accept=true`
candidates after its own safety gates.

After any rollback, the accepted masks and accepted point-session history remain
authoritative; the rejected candidate is retained only for audit and replan
context. Direct does not reuse the rejected `suggested_action`: it makes a new
full-context Qwen call with the rejected action, mask delta, and rejection
reason, plus a bounded four-entry history of recent rejected actions. The
replan contract uses `candidate_effect=null`; runtime assigns
`comparison="uncertain"` and `accept=false`, while Qwen supplies a different
executable action or a `finish`.

For the post-fix Direct-only three-sample run, first probe a chosen GPU node,
then pin the run to that same node:

```bash
NODE=gpuXX BAILIAN_NETWORK_MODE=direct \
  bash tools/submit_ca0721_direct_replan.sh 9
```

The run writes to `outputs/CA_0721(9)-bailian-direct-rollback-replan/`, with
Slurm stdout/stderr archived below its `logs/` directory.
