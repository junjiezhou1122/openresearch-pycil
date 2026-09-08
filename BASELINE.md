# Baseline provenance

Status: **baseline_certified** (verifier PASS on 2026-09-08)

- Upstream: `https://github.com/LAMDA-CL/PyCIL.git`
- Upstream commit: `f3509b8ca3f20660ce4aa13f19d5283de81b4b35`
- FML-bench source: `https://github.com/qrzou/FML-bench.git`
- FML-bench commit: `d336651ebea50c622c256f02ded82b68b4451fdc`
- FML task: `Continual_Learning_pycil`
- FML evaluator: `train_eval_baseline.py`, sha256 `2e58757a72cd70432a50d9c97f690958ca3cf68e767a7ccd411942eb20b6c3b6` (fetched and hash-checked per run by `public_validation/run.sh`)

## Certification record

- Frozen protocol: `protocol/baseline-protocol.md` (`fml-lite-pycil-baseline-v1`, frozen before measurement)
- Certified commit: `15c78044b94d57c0cfd84eedc6b547b3c87b99d9` (tag `baseline/v1`)
- Base commit (imported starter): `a7b941d468c10cee09f0ed26f9a47ec2b47da7f8`
- Verified diff base→certified: `public_validation/run.sh`, `protocol/baseline-protocol.md` only; `algorithm.py` byte-identical to the starter (verifier check `algorithm_unchanged`)
- Runner: OpenResearch `ssh-bundle` backend, fresh detached checkout per run, exact-SHA dispatch
- Command: `bash public_validation/run.sh val`

## Runs (all exit 0, commit checkout verified, evaluator hash verified)

| job_id | avg_incremental_acc_mean | avg_acc_cnn_mean | duration |
|---|---|---|---|
| pycil-5ea1f7030bea | 0.59485 | 0.5361667 | 1131.8 s |
| pycil-5162a9a9d903 | 0.59485 | 0.5361667 | 1098.9 s |
| pycil-0c64cead84a1 | 0.59485 | 0.5361667 | 1105.5 s |

- Metric mean: **0.59485** (`avg_incremental_acc_mean`, NME, higher is better; val split = 30% of CIFAR-100 test, split seed 42)
- Run-to-run spread: 0.0 pp (cudnn deterministic; per protocol §11.7 limit is 1.0 pp)
- Environment: `fuxin` / Ubuntu 20.04.6 / Python 3.8.10 / torch 2.4.1+cu121 / driver 550.78 / 1× RTX 3090 (`CUDA_VISIBLE_DEVICES=0`), dependency lock sha256 `f8ae96df0beaac5d6d98a25290c41e970b1ae56cc9ca30aa575267ce10b28a4c`, dataset sha256 `85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7` (md5 `eb9058c3a382ffc7106e4002c42a8d85`, upstream-canonical)
- Independent verifier: `openresearch.certify` (OpenResearch repo, separate process), machine-readable verdict `{"decision": "PASS"}`, evidence_sha256 `bcea685a3d692de4c2028236e690a11a559a56296be67125483ff810afa9408d`, 44/44 checks passed
- Evidence bundle: `.openresearch/artifacts/<job-id>/` on the OpenResearch control plane; auditable copy committed under `certification/pycil-baseline-v1/` in the OpenResearch repository

Reference: FML-bench published baseline for this task is 0.59107 (val) / 0.59517 (test); the certified value on the val split is within expected range.

The 70% hidden test split was NOT evaluated; holdout verification is reserved for the independent evaluator and for candidate promotion later.
