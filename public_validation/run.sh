#!/usr/bin/env bash
# FML-bench-Lite PyCIL baseline/candidate execution entry point.
#
# Executed by the OpenResearch control plane inside a fresh detached checkout of
# an exact commit SHA. Every failure must propagate (no `|| true`, no silent
# retries). This script is harness infrastructure; the only worker-modifiable
# file is algorithm.py (see protocol/baseline-protocol.md).
#
# Usage: bash public_validation/run.sh {val|test}
set -euo pipefail

SPLIT="${1:?usage: run.sh {val|test}}"
case "$SPLIT" in val|test) ;; *) echo "unknown split: $SPLIT" >&2; exit 2;; esac

FROZEN_VENV="${FROZEN_VENV:-/home/zhoujunjie/openresearch-envs/pycil-baseline}"
DATASET_DIR="${DATASET_DIR:-/home/zhoujunjie/openresearch-data}"
EVALUATOR_URL="https://raw.githubusercontent.com/qrzou/FML-bench/d336651ebea50c622c256f02ded82b68b4451fdc/ml_tasks/Continual_Learning_pycil/train_eval_baseline.py"
EVALUATOR_SHA256="2e58757a72cd70432a50d9c97f690958ca3cf68e767a7ccd411942eb20b6c3b6"
CIFAR100_TGZ_MD5="eb9058c3a382ffc7106e4002c42a8d85"

PY="$FROZEN_VENV/bin/python"
[ -x "$PY" ] || { echo "FATAL: frozen venv python missing: $PY" >&2; exit 3; }

# --- GPU pinning (protocol §6): single healthy RTX 3090 ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONHASHSEED=0

# --- dataset cache with identity check (fail fast on mismatch) ---
mkdir -p data
if [ ! -f "data/cifar-100-python.tar.gz" ]; then
    cp "$DATASET_DIR/cifar-100-python.tar.gz" "data/cifar-100-python.tar.gz"
fi
ACTUAL_MD5=$(md5sum "data/cifar-100-python.tar.gz" | cut -d' ' -f1)
if [ "$ACTUAL_MD5" != "$CIFAR100_TGZ_MD5" ]; then
    echo "FATAL: dataset md5 mismatch: got $ACTUAL_MD5 want $CIFAR100_TGZ_MD5" >&2
    exit 4
fi
if [ ! -d "data/cifar-100-python" ]; then
    tar -xzf "data/cifar-100-python.tar.gz" -C data
fi

# --- evaluator: fetch from pinned FML commit and verify identity ---
if [ -f "train_eval_baseline.py" ]; then
    ACTUAL_EVAL_SHA=$(sha256sum train_eval_baseline.py | cut -d' ' -f1)
    if [ "$ACTUAL_EVAL_SHA" != "$EVALUATOR_SHA256" ]; then
        echo "FATAL: evaluator hash mismatch: got $ACTUAL_EVAL_SHA want $EVALUATOR_SHA256" >&2
        exit 5
    fi
else
    curl -s --max-time 60 -o train_eval_baseline.py "$EVALUATOR_URL"
    ACTUAL_EVAL_SHA=$(sha256sum train_eval_baseline.py | cut -d' ' -f1)
    if [ "$ACTUAL_EVAL_SHA" != "$EVALUATOR_SHA256" ]; then
        echo "FATAL: evaluator download hash mismatch: got $ACTUAL_EVAL_SHA want $EVALUATOR_SHA256" >&2
        exit 5
    fi
fi

# --- environment identity (single-line JSON on stdout for the evidence bundle) ---
ENV_IDENTITY_JSON=$("$PY" - <<'PYEOF'
import hashlib, json, platform, subprocess, sys
identity = {
    "hostname": platform.node(),
    "os_release": "",
    "kernel": platform.release(),
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
}
try:
    identity["os_release"] = open("/etc/os-release").read().split("PRETTY_NAME=")[1].split("\n")[0].strip('"')
except Exception as exc:
    identity["os_release"] = f"unreadable: {exc}"
import torch
identity["torch"] = torch.__version__
identity["torch_cuda"] = torch.version.cuda
identity["cudnn"] = torch.backends.cudnn.version()
identity["cuda_available"] = torch.cuda.is_available()
identity["gpu_count"] = torch.cuda.device_count()
if torch.cuda.is_available():
    identity["gpu_name"] = torch.cuda.get_device_name(0)
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
identity["dataset_tar_gz_sha256"] = sha256_file("data/cifar-100-python.tar.gz")
identity["evaluator_sha256"] = sha256_file("train_eval_baseline.py")
lock = "/home/zhoujunjie/openresearch-envs/requirements-lock.txt"
identity["requirements_lock_sha256"] = sha256_file(lock)
probe = subprocess.run(
    ["nvidia-smi", "--id=0", "--query-gpu=index,name,driver_version,memory.total",
     "--format=csv,noheader"], capture_output=True, text=True)
identity["nvidia_smi_gpu0"] = probe.stdout.strip() or f"exit {probe.returncode}: {probe.stderr.strip()}"
print("ENV_IDENTITY_JSON " + json.dumps(identity, sort_keys=True))
PYEOF
)
echo "$ENV_IDENTITY_JSON"

# --- training + evaluation (metric of record) ---
rm -rf results_tmp
"$PY" train_eval_baseline.py --split "$SPLIT"

# --- emit machine-readable result (single line, evidence of record) ---
"$PY" - "$SPLIT" <<'PYEOF'
import json, pathlib, sys
info = json.loads(pathlib.Path(f"results_tmp/{sys.argv[1]}_info.json").read_text())
means = info["cifar100_incremental"]["means"]
print("BASELINE_RESULT_JSON " + json.dumps({
    "schema": "openresearch.baseline-result.v1",
    "metric": "avg_incremental_acc_mean",
    "value": means["avg_incremental_acc_mean"],
    "avg_acc_cnn_mean": means["avg_acc_cnn_mean"],
    "avg_acc_nme_mean": means["avg_acc_nme_mean"],
    "source": f"results_tmp/{sys.argv[1]}_info.json",
}, sort_keys=True))
PYEOF
