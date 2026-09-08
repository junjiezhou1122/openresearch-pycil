# FML-bench-Lite PyCIL Baseline Protocol (frozen)

Protocol ID: `fml-lite-pycil-baseline-v1`
Frozen at: 2026-09-08 (commit that introduced this file)
Status: frozen BEFORE any baseline measurement was observed. Acceptance criteria
in §9 must not be edited after baseline results are seen; any later change
requires a new protocol ID and re-certification.

## 1. Upstream provenance

| Item | Value |
|---|---|
| Upstream repository | https://github.com/LAMDA-CL/PyCIL.git |
| Upstream snapshot commit | `f3509b8ca3f20660ce4aa13f19d5283de81b4b35` (tag `import/upstream-f3509b8ca3f2`) |
| FML-bench source | https://github.com/qrzou/FML-bench.git |
| FML-bench commit | `d336651ebea50c622c256f02ded82b68b4451fdc` |
| FML task | `ml_tasks/Continual_Learning_pycil` |
| FML adapter file | `algorithm.py` (byte-identical to FML source) |
| FML evaluator script | `train_eval_baseline.py`, sha256 `2e58757a72cd70432a50d9c97f690958ca3cf68e767a7ccd411942eb20b6c3b6` |

The evaluator script is NOT committed into this repository. For every certified
run the control plane downloads it from the FML-bench repository at the pinned
commit and records its sha256 in the run evidence. If the downloaded hash does
not match the pinned hash, the run aborts (fail fast).

## 2. Data

| Item | Value |
|---|---|
| Dataset | CIFAR-100 (Krizhevsky python version) |
| Source | https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz |
| Identity check | md5 `eb9058c3a382ffc7106e4002c42a8d85` (canonical, upstream-published) |
| Train/test boundary | canonical: 50000 train / 10000 test |
| Val/test split | CIFAR-100 test set stratified per class: 30% val (3000), 70% test (7000) |
| Split seed | `split_seed=42`, `numpy.random.RandomState(42)`, per-class permutation (evaluator-owned, not modifiable) |
| Training data split | PyCIL DataManager with `shuffle=True, seed=1993` (class-then-sample ordering, evaluator-owned) |

The worker never touches the split logic. Val and test evaluate the same trained
model; only the evaluation subset differs.

## 3. Model and training configuration (baseline)

From `algorithm.py::get_pycil_config()` (the only worker-modifiable file):

```json
{
  "prefix": "benchmark",
  "dataset": "cifar100",
  "memory_size": 2000,
  "memory_per_class": 20,
  "fixed_memory": false,
  "shuffle": true,
  "init_cls": 50,
  "increment": 10,
  "model_name": "icarl",
  "convnet_type": "resnet32",
  "device": ["0"],
  "seed": [1993],
  "init_epoch": 60, "init_lr": 0.1, "init_milestones": [20, 40, 50],
  "init_lr_decay": 0.1, "init_weight_decay": 0.0005,
  "epochs": 50, "lrate": 0.1, "milestones": [20, 35],
  "lrate_decay": 0.1, "batch_size": 128, "weight_decay": 0.0002
}
```

Schedule: 50 base classes, then 5 stages × 10 classes (100 total). iCaRL with
knowledge distillation and NME classification.

## 4. Seeds

- Task/data seed: `1993` (fixed, in config).
- Split seed: `42` (fixed, in evaluator).
- PyCIL internal `_set_random()` seeds torch with 1 (upstream code, unchanged).

Multi-run variance is therefore dominated by CUDA nondeterminism
(`cudnn.benchmark=False`, `deterministic=True` are set, but GPU reductions are
not fully bitwise-stable across kernel schedules under concurrent load).

## 5. Metric

- Primary: `avg_incremental_acc_mean` (higher is better) = mean top-1 accuracy
  across all incremental stages, NME classifier for iCaRL
  (falls back to CNN if NME absent).
- Parsed by the FML evaluator from the trainer's `Average Accuracy (NME|CNN):`
  stdout lines. Reported on the val subset during development; the hidden test
  subset (70%) is reserved for holdout verification and is NOT evaluated for
  baseline certification.

## 6. Compute environment (hangzhou_server)

| Item | Value |
|---|---|
| Host alias | `hangzhou_server` (ssh), hostname `fuxin` |
| OS | Ubuntu 20.04, kernel 5.15.0-139-generic, x86_64 |
| CPU/RAM | 80 cores, 377 GB RAM |
| GPU | NVIDIA GeForce RTX 3090 24GB ×8 (GPU 7 currently failed at PCI level); baseline pinned to `CUDA_VISIBLE_DEVICES=0` |
| Driver | 550.78 (kernel module) |
| CUDA runtime | 12.4 toolkit at /usr/local/cuda-12.4; PyTorch wheel bundles CUDA 12.1 runtime |
| Python | 3.8.10 (`/usr/bin/python3`) in dedicated venv `/home/zhoujunjie/openresearch-envs/pycil-baseline` |
| Dependency lock | `requirements-lock.txt` on the server; sha256 recorded in `environment-identity.json` |
| Dataset cache | `/home/zhoujunjie/openresearch-data/cifar-100-python.tar.gz` (md5 recorded in `environment-identity.json`) |
| Transport | `ssh-bundle` (server → GitHub is unreliable: TLS resets/timeouts observed 2026-09-08) |

## 7. Resource and time budget

- Wall-clock per run: ≤ 120 minutes (runner timeout 7200 s). Amended from 90
  minutes BEFORE any certified measurement was observed, based on a 1-epoch
  smoke estimate and server CPU contention (shared host, load ~90).
- GPU memory: single 3090 (24 GB) is sufficient for ResNet-32 batch 128.
- The run script aborts (non-zero exit) if the metric cannot be parsed.

## 8. Runner contract

Entry point committed in this repository: `public_validation/run.sh`
- bootstraps the frozen venv, dataset cache, pinned evaluator;
- executes `python train_eval_baseline.py --split val`;
- prints an `ENV_IDENTITY_JSON` line and a `BASELINE_RESULT_JSON` line on stdout;
- propagates every failure (no `|| true`, no silent retries).

The OpenResearch control plane dispatches the exact 40-char commit SHA via
`ssh-bundle`, checks it out detached in a fresh job directory, runs the command,
and preserves request.json / completion.json / stdout.log / stderr.log / exit
code / runtime identity in an append-only local ledger.

## 9. Worker-modifiable surface

- `algorithm.py` (per `.openresearch/task.json` `worker_edit_paths`).
- Everything else (trainer, models/, utils/, evaluator, splits, run harness) is
  evaluator-owned. Diff outside `algorithm.py` is only allowed for this
  certification branch's harness files (`public_validation/`, protocol docs),
  which are frozen before measurement and identical across all certified runs.

## 10. Evaluator command (frozen)

```
python train_eval_baseline.py --split val
```

 executed from the checkout root with the frozen venv python, after the pinned
 evaluator file has been fetched and hash-checked by `public_validation/run.sh`.

## 11. baseline/v1 promotion criteria (frozen before measurement)

Baseline certification requires ALL of:

1. Runner status `succeeded` with `exit_code == 0` for **3 independent runs** on
   the exact same commit SHA (fresh checkout each run via ssh-bundle).
2. Each run's stdout contains a parseable `BASELINE_RESULT_JSON` with
   `metric == avg_incremental_acc_mean` and a finite float value.
3. Requested commit == checked-out commit in every completion.json.
4. No diff outside the allowed files between the certified commit and its
   parent, verified by the independent verifier.
5. Evaluator file hash in every run matches the pinned sha256 (§1).
6. Environment identity recorded and hash-anchored (§6), same venv across runs.
7. Run-to-run spread of the primary metric: `max - min <= 1.0` percentage
   points. If spread exceeds this, report variance honestly; do NOT relax the
   criterion within this protocol. (3+ additional runs may be added under the
   same frozen criteria to characterize the distribution.)
8. An independent verifier (separate process, not the worker, runner, or goal
   model) checks items 1–7 plus evidence-bundle hash integrity and emits a
   machine-readable verdict `{"decision": "PASS"}` or `{"decision": "FAIL", ...}`.
9. Only after the verifier PASS: create tag `baseline/v1` on the certified
   commit, update task status to `baseline_certified`, and update BASELINE.md.

The certified baseline value is the mean of the 3 primary-metric values; the
per-run values are recorded unmodified in the evidence bundle.
