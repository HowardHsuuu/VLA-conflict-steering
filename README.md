# Cross-Modal Conflict Monitoring and Steering in VLAs

This repository studies a deployment failure mode in which a robot instruction
conflicts with the current scene. A probe reads scene evidence from a frozen VLA,
compares it with the instruction claim, and gates a task-configured correction in
the action expert's flow trajectory. The action prompt remains unchanged during
action-flow steering.

[Project page](https://vla-conflict-steering.howardhsu.chatgpt.site/) ·
[Frozen protocol](configs/cross_task_decisive_v1.toml) ·
[Public result release](results/cross_task_decisive_v1/manifest.json) ·
[Qualitative rollouts](assets/rollouts/cross_task_decisive_gallery_v1/manifest.json)

## Main result

The frozen evaluation contains 660 rollouts across five configured tasks, four
unseen scenes per task, three shared action-flow noise seeds, and eleven matched
conditions.

| Condition | Success |
| --- | ---: |
| Conflict, unsteered | 8.3% (5/60) |
| Action-flow steering, learned trigger | **75.0% (45/60)** |
| Action-flow steering, oracle trigger | 75.0% (45/60) |
| Action-flow steering, always on | 76.7% (46/60) |
| Prompt correction, learned trigger | 81.7% (49/60) |
| Aligned instruction, unsteered | 76.7% (46/60) |
| Aligned instruction, learned monitor | 76.7% (46/60) |
| Aligned instruction, forced steering | 15.0% (9/60) |
| Wrong-sign steering | 1.7% (1/60) |
| Norm-matched random steering | 13.3% (8/60) |

Learned-triggered action-flow steering yields 40 paired wins, zero losses, and
20 ties against the unsteered conflict condition. The monitor triggers in 60/60
conflict rollouts and 0/60 aligned rollouts. All five task effects are positive;
the task-stratified scene-cluster bootstrap interval for the pooled gain is
[+55.0, +78.3] percentage points. All ten preregistered gates pass.

Under the same monitor information, prompt correction succeeds in 49/60 trials
and action-flow steering in 45/60. We therefore treat action-flow steering as a
non-language intervention surface for settings where the instruction channel is
immutable, not as a replacement for prompt correction.

## Qualitative rollouts

The release includes three matched failure/recovery pairs replayed from cells in
the frozen evaluation: one absent-object conflict and two spatial conflicts. Each
pair fixes the scene, conflicting instruction, simulator state, and action-flow
noise while changing only whether the learned monitor may trigger action-flow
steering. All six MP4s reproduce their recorded outcomes exactly and include
hash-bound manifests.

These examples are outcome-selected mechanism illustrations, not new trials or
representative samples. The aggregate result above remains the efficacy evidence.

## Method

1. A neutral diagnostic query elicits a residual representation of the observed
   scene state.
2. A calibrated probe maps that representation to task-relevant visual evidence.
3. The monitor compares visual evidence with the claim encoded by the instruction.
4. On a confident conflict, a task-specific controller shifts the action expert's
   denoising trajectory; task-specific release logic stops the intervention.

The offline reference instruction is used to fit each configured controller. It
is not queried during action-flow steering. This is configured-task procedure
generalization: a new task may use the same fitting and evaluation procedure with
its own small offline calibration set. It is not zero-shot controller transfer.

## Verify the public result table

The verifier requires no simulator, model checkpoint, or accelerator. It checks
the released aggregate table, its internal arithmetic, the reported paired
comparisons, monitor triggers, gates, and the declared protocol inventory.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/causal-vla-results
.venv/bin/pytest -q
```

Expected headline: `Validated aggregate for 660 rollouts; all 10 preregistered gates pass.`

This Git repository intentionally excludes raw per-rollout payloads, fitted
controller weights, and machine-specific execution receipts. Those high-entropy
artifacts are not necessary to inspect the method or validate the public result
table.

## Reproduce the simulator campaign

Full simulator reproduction additionally requires the fitted task controllers,
Action Atlas at commit
`b8b0db331df18fc30a3fd92c45ec721d35d3ee52`, its LIBERO and LeRobot submodules,
and SmolVLA-LIBERO revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`.
The macOS environment is locked in `environments/smolvla-macos/uv.lock`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  environments/smolvla-macos/.venv/bin/python \
  -m causal_vla.cross_task_decisive_campaign \
  --output artifacts/results/cross_task_decisive_v1_run

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  environments/smolvla-macos/.venv/bin/python \
  -m causal_vla.cross_task_decisive_analysis \
  --output artifacts/results/cross_task_decisive_v1_run
```

The evaluated campaign encountered cumulative MPS memory growth after 175 cells.
It was continued with outcome-blind process isolation while retaining those 175
cells byte-for-byte. This is one frozen campaign, not two independent
confirmations; the aggregate release records that boundary without publishing
the machine-specific receipt.

## Scope

The evidence supports a repeatable task-configuration procedure on one SmolVLA
checkpoint in LIBERO simulation. It does not establish universal conflict
detection, zero-shot transfer, real-robot safety, or superiority to replanning or
prompt correction.

The rollout videos in `assets/rollouts/` are explicitly outcome-selected
mechanism illustrations. The three-pair gallery replays cells from the frozen
660-rollout evaluation; the retained `spatial_scene10_noise457` pair comes from
the earlier development study. Neither is a random sample.

## License

Original code is released under the [Apache License 2.0](LICENSE). SmolVLA,
Action Atlas, LeRobot, LIBERO, and their assets remain under their upstream terms.
