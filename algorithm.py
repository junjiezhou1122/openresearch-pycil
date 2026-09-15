"""
Baseline continual learning algorithm: iCaRL on CIFAR-100.

This file contains the epoch and hyperparameter configuration for iCaRL.
The actual iCaRL implementation is in the PyCIL repository (models/icarl.py).

Agents should modify the hyperparameters below, or replace the model_name
with a different continual learning algorithm from PyCIL (e.g., 'der', 'foster',
'memo', 'ewc', 'lwf', etc.).

To make deeper changes, agents can also modify models/icarl.py directly,
but must keep the BaseLearner interface (incremental_train, eval_task, after_task).

-------------------------------------------------------------------------------
H025 readout-geometry probe at the 6000 budget (does the budget SUBSUME the
angular readout lever?)
-------------------------------------------------------------------------------
Configuration is the H018 stack: iCaRL + ResNet-18, init_epoch=100, epochs=50,
seed 1993 (harness-pinned), memory_size = 6000. The intervention is H008's, unchanged:
an EVALUATION-ONLY monkeypatch of BaseLearner._eval_cnn (inherited by iCaRL) so the
CNN head scores by cosine similarity - L2-normalized features dotted with
L2-normalized fc.weight rows, no bias - with the unmodified production affine head
recomputed in the same pass as an in-run control. Training is byte-identical to H018.

Why this experiment. Two independent levers in this loop have moved the CNN channel:
  * H008 (valid/supported) showed that at the 2000 budget the affine head's new-class
    weight-norm advantage (~25-35% larger norms) is causal: switching to cosine at
    evaluation raised aggregate CNN 0.5928 -> 0.6466833 (+5.3883 pp) with NME
    bit-identical and the in-run affine recomputation reproducing H001 exactly.
  * The memory budget (H013/H018) raised the CNN channel too: 59.28 -> 64.59 -> 68.39
    as memory went 2000 -> 4000 -> 6000.
Those two facts raise an interaction question the loop has never tested: is the
classifier-norm distortion still present once the learner stores 6000 exemplars, or
has the enlarged replay set already cured it, leaving the angular readout with little
left to fix? If the budget SUBSUMES the readout lever, the cosine gain at 6000 should
be materially smaller than its 5.3883 pp at 2000.

Hypothesis (subsumption). At the 6000 budget the cosine readout's aggregate CNN gain
over the affine head is at most HALF its 2000 value, i.e. <= 2.6942 pp. If the gain
remains above that, the two levers are independent/additive rather than substitutable.

Pre-registered reading: SUPPORTED if (cosine - affine) aggregate CNN gain at 6000 is
<= +2.6942 pp; REFUTED if it is larger.

Scope and controls (all exact, because the intervention is evaluation-only):
  * NME must be bit-identical to H018 at every stage - the cosine patch cannot touch
    the NME path - which is a training-identity control.
  * The in-run affine recomputation on the same features must reproduce H018's CNN
    per-stage totals (80.67/71.78/69.86/64.88/62.67/60.5), proving the features and
    classifier weights are exactly H018's.
  * Persistent state and the CPU/CUDA RNG stream are never written; sha256 checks are
    printed per stage.
"""

import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base import BaseLearner
from utils.toolkit import accuracy

_EPS = 1e-8


def _h025_cosine_eval_cnn(self, loader):
    """H025: rank classes by normalized-feature dot normalized-classifier-weight.

    Returns ``(y_pred, y_true)`` with the same contract as
    ``BaseLearner._eval_cnn`` ([N, topk] int array, [N] int array), so
    ``eval_task`` / ``_evaluate`` consume it unchanged. The cosine scores carry
    no bias term; the affine head is recomputed inside the same pass purely as
    an in-run control and diagnostic.
    """
    net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
    net.eval()

    nb_old = int(getattr(self, "_known_classes", 0))
    topk = self.topk

    weight = net.fc.weight.detach()
    bias = None if net.fc.bias is None else net.fc.bias.detach()
    weight_unit = F.normalize(weight, dim=1, eps=_EPS)

    cosine_preds, affine_preds, y_true = [], [], []
    feat_norm_sum, feat_norm_cnt = {}, {}

    for _, (_, inputs, targets) in enumerate(loader):
        inputs = inputs.to(self._device)
        with torch.no_grad():
            features = net.extract_vector(inputs)
            cosine_logits = F.normalize(features, dim=1, eps=_EPS) @ weight_unit.t()
            affine_logits = F.linear(features, weight, bias)

        cosine_preds.append(
            torch.topk(cosine_logits, k=topk, dim=1, largest=True, sorted=True)[1].cpu().numpy()
        )
        affine_preds.append(
            torch.topk(affine_logits, k=topk, dim=1, largest=True, sorted=True)[1].cpu().numpy()
        )
        y_true.append(targets.cpu().numpy())

        batch_norms = features.norm(p=2, dim=1)
        for class_id in targets.unique().tolist():
            mask = targets == class_id
            feat_norm_sum[class_id] = feat_norm_sum.get(class_id, 0.0) + float(
                batch_norms[mask].sum()
            )
            feat_norm_cnt[class_id] = feat_norm_cnt.get(class_id, 0) + int(mask.sum())

    y_pred = np.concatenate(cosine_preds)
    y_true = np.concatenate(y_true)

    # ---- diagnostics only (never used for the returned prediction) ----
    feat_norm_by_class = {
        str(c): feat_norm_sum[c] / feat_norm_cnt[c] for c in sorted(feat_norm_sum)
    }
    weight_norm_by_class = [round(float(v), 6) for v in weight.norm(p=2, dim=1)]
    bias_by_class = None if bias is None else [round(float(v), 6) for v in bias]

    def _old_new_split(values):
        old = [v for i, v in enumerate(values) if i < nb_old]
        new = [v for i, v in enumerate(values) if i >= nb_old]
        return (
            float(np.mean(old)) if old else 0.0,
            float(np.mean(new)) if new else 0.0,
        )

    feat_norm_old, feat_norm_new = _old_new_split(
        [feat_norm_by_class[str(c)] for c in sorted(feat_norm_by_class)]
    )
    weight_norm_old, weight_norm_new = _old_new_split(weight_norm_by_class)
    if bias_by_class is None:
        bias_abs_old, bias_abs_new = 0.0, 0.0
    else:
        bias_abs_old, bias_abs_new = _old_new_split([abs(v) for v in bias_by_class])

    print(
        "H025_DIAG "
        + json.dumps(
            {
                "readout": "cosine (L2 features . L2 fc.weight, bias ignored)",
                "nb_old": nb_old,
                "nb_total": int(weight.shape[0]),
                "cnn_cosine_grouped": accuracy(y_pred.T[0], y_true, nb_old),
                "cnn_affine_grouped_in_run_control": accuracy(
                    np.concatenate(affine_preds).T[0], y_true, nb_old
                ),
                "feature_norm_mean": float(
                    sum(feat_norm_sum.values()) / max(sum(feat_norm_cnt.values()), 1)
                ),
                "feature_norm_by_class_mean": {
                    c: round(v, 6) for c, v in feat_norm_by_class.items()
                },
                "feature_norm_old_mean": feat_norm_old,
                "feature_norm_new_mean": feat_norm_new,
                "fc_weight_norm_by_class": weight_norm_by_class,
                "fc_weight_norm_old_mean": weight_norm_old,
                "fc_weight_norm_new_mean": weight_norm_new,
                "fc_bias_by_class": bias_by_class,
                "fc_bias_abs_old_mean": bias_abs_old,
                "fc_bias_abs_new_mean": bias_abs_new,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    return y_pred, y_true


def _install_h025_cosine_readout():
    """Install the cosine CNN readout exactly once (no-op if already installed)."""
    if getattr(BaseLearner._eval_cnn, "_h025_cosine_readout", False):
        return False
    _h025_cosine_eval_cnn._h025_cosine_readout = True
    BaseLearner._eval_cnn = _h025_cosine_eval_cnn
    return True


_H025_PATCH_INSTALLED = _install_h025_cosine_readout()
print(
    "H025_PATCH_INSTALLED "
    + json.dumps(
        {
            "installed": _H025_PATCH_INSTALLED,
            "target": "BaseLearner._eval_cnn (inherited by iCaRL)",
            "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
            "readout": "cosine (L2 features . L2 fc.weight, bias ignored)",
            "training_untouched": True,
        },
        sort_keys=True,
    ),
    flush=True,
)


def get_pycil_config():
    """
    Return PyCIL experiment configuration.

    Agents can modify:
    - model_name: Switch to a different CL algorithm (der, foster, memo, etc.)
    - init_epoch / epochs: Number of training epochs per stage
    - memory_size: Total exemplar memory budget
    - init_cls / increment: Class-incremental schedule
    - convnet_type: Backbone architecture
    - Other hyperparameters specific to the chosen model
    """
    return {
        "prefix": "benchmark",
        "dataset": "cifar100",
        "memory_size": 6000,   # H025: H018 stack (H008 ran at 2000)
        "memory_per_class": 20,
        "fixed_memory": False,
        "shuffle": True,
        "init_cls": 50,
        "increment": 10,
        "model_name": "icarl",
        "convnet_type": "resnet18",
        "device": ["0"],
        "seed": [1993],

        # Reduced epoch settings for 30-min budget
        # Original: init_epoch=200, epochs=170
        # Reduced: init_epoch=60, epochs=50
        "init_epoch": 100,
        "init_lr": 0.1,
        "init_milestones": [30, 70, 90],
        "init_lr_decay": 0.1,
        "init_weight_decay": 0.0005,

        "epochs": 50,
        "lrate": 0.1,
        "milestones": [20, 35],
        "lrate_decay": 0.1,
        "batch_size": 128,
        "weight_decay": 0.0002,
    }
