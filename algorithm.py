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
H007b evaluation-only classifier weight-norm alignment (readout-isolated repair)
-------------------------------------------------------------------------------
Training is byte-identical to the verified H001 configuration (iCaRL +
ResNet-18, init_epoch=100, epochs=50, seed 1993, memory_size=2000). The single
change is an *evaluation-time* monkeypatch of ``BaseLearner._eval_cnn``
(inherited by ``iCaRL``): the CNN head ranks classes using logits computed from a
**copy** of ``fc.weight`` whose new-class rows have been rescaled so their mean
L2 norm matches the old-class mean row norm. That is exactly the H007
calibration, applied to a temporary tensor instead of the live parameter.

Why a copy (the H007 defect being repaired): H007 called
``IncrementalNet.weight_align`` on the persistent network *before* ``eval_task``.
That mutated the classifier, and ``after_task`` then froze the aligned network as
the next stage's KD teacher, so NME drifted and the observed CNN effect was
confounded with a training change (invalid experiment). Here the alignment lives
only inside ``_eval_cnn``: ``fc.weight``, ``fc.bias``, the convnet,
``_class_means``, the exemplar memory and every training step are untouched, so
all downstream state must stay H001-identical.

Scope / auditability:
  * ``gamma = mean(||W_old rows||) / mean(||W_new rows||)`` and ``W_new *= gamma``
    over exactly the newest ``_total_classes - _known_classes`` rows -- the same
    definition and the same rows as ``IncrementalNet.weight_align``. The base
    stage (no old classes) is left unaligned and reported as such.
  * The unaligned head is the unmodified production forward
    ``net(inputs)["logits"]`` from the same batch loop; the aligned head is one
    extra ``F.linear`` on those already-computed features. No extra DataLoader
    iteration and no RNG consumption, so the training stream cannot shift (the
    H009 failure mode).
  * sha256 checksums of ``fc.weight``, ``fc.bias``, the whole ``state_dict``,
    ``_class_means`` and the exemplar memory are compared before/after every
    ``_eval_cnn`` call and every ``eval_task`` call and printed per stage. Any
    mismatch is printed and raised (fail fast), because a persistent change would
    invalidate the experiment.
  * One machine-readable ``H007B_DIAG`` line per stage carries gamma, the
    aligned and unaligned grouped accuracies and the invariance checksums.
"""

import hashlib
import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.base import BaseLearner
from utils.toolkit import accuracy


def _h007b_sha256_tensor(tensor):
    """Content hash of one tensor (device-independent, no aliasing)."""
    array = tensor.detach().to("cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _h007b_sha256_network(net):
    """Content hash of a whole network state_dict (convnet + classifier)."""
    digest = hashlib.sha256()
    state = net.state_dict()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(_h007b_sha256_tensor(state[name]).encode("ascii"))
    return digest.hexdigest()


def _h007b_sha256_array(value):
    """Content hash of a numpy-like attribute (e.g. ``_class_means``)."""
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _h007b_align_new_class_rows(weight, nb_old, nb_total):
    """Return ``(aligned_copy, gamma, applied)``; never mutates ``weight``.

    ``gamma`` and the rescaled rows replicate ``IncrementalNet.weight_align``
    exactly (same mean-norm ratio, same trailing ``increment`` rows), but the
    scaling is applied to a detached clone owned by the caller.
    """
    if nb_old <= 0 or nb_old >= nb_total:
        return weight.detach().clone(), 1.0, False
    aligned = weight.detach().clone()
    new_norm = aligned[nb_old:, :].norm(p=2, dim=1)
    old_norm = aligned[:nb_old, :].norm(p=2, dim=1)
    gamma = float(old_norm.mean()) / float(new_norm.mean())
    aligned[nb_old:, :] *= gamma
    return aligned, gamma, True


def _h007b_eval_cnn(self, loader):
    """H007b: eval-only aligned CNN readout from copied classifier weights.

    Returns ``(y_pred, y_true)`` with the same contract as
    ``BaseLearner._eval_cnn`` ([N, topk] int array, [N] int array) so
    ``eval_task`` / ``_evaluate`` consume it unchanged. The prediction returned
    to the harness comes from the *aligned copy*; the persistent parameters are
    only read, and are proven unchanged by before/after content hashes.
    """
    net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
    net.eval()

    nb_old = int(getattr(self, "_known_classes", 0))
    topk = int(self.topk)

    weight = net.fc.weight.detach()
    bias = None if net.fc.bias is None else net.fc.bias.detach()
    nb_total = int(weight.shape[0])

    aligned_weight, gamma, applied = _h007b_align_new_class_rows(weight, nb_old, nb_total)
    if applied:
        old_norm_mean = float(weight[:nb_old, :].norm(p=2, dim=1).mean())
        new_norm_mean = float(weight[nb_old:, :].norm(p=2, dim=1).mean())
        aligned_new_norm_mean = float(aligned_weight[nb_old:, :].norm(p=2, dim=1).mean())
    else:
        old_norm_mean = 0.0
        new_norm_mean = 0.0
        aligned_new_norm_mean = 0.0

    fc_weight_sha_before = _h007b_sha256_tensor(weight)
    fc_bias_sha_before = None if bias is None else _h007b_sha256_tensor(bias)
    state_sha_before = _h007b_sha256_network(net)
    class_means_sha_before = _h007b_sha256_array(getattr(self, "_class_means", None))

    aligned_preds, unaligned_preds, y_true = [], [], []
    for _, (_, inputs, targets) in enumerate(loader):
        inputs = inputs.to(self._device)
        with torch.no_grad():
            outputs = net(inputs)  # unmodified production forward (unaligned control)
            unaligned_logits = outputs["logits"]
            aligned_logits = F.linear(outputs["features"], aligned_weight, bias)
        aligned_preds.append(
            torch.topk(aligned_logits, k=topk, dim=1, largest=True, sorted=True)[1].cpu().numpy()
        )
        unaligned_preds.append(
            torch.topk(unaligned_logits, k=topk, dim=1, largest=True, sorted=True)[1].cpu().numpy()
        )
        y_true.append(targets.cpu().numpy())

    aligned_preds = np.concatenate(aligned_preds)
    unaligned_preds = np.concatenate(unaligned_preds)
    y_true = np.concatenate(y_true)

    fc_weight_sha_after = _h007b_sha256_tensor(net.fc.weight.detach())
    fc_bias_sha_after = None if net.fc.bias is None else _h007b_sha256_tensor(net.fc.bias.detach())
    state_sha_after = _h007b_sha256_network(net)
    class_means_sha_after = _h007b_sha256_array(getattr(self, "_class_means", None))

    persistent_unchanged = bool(
        fc_weight_sha_before == fc_weight_sha_after
        and fc_bias_sha_before == fc_bias_sha_after
        and state_sha_before == state_sha_after
        and class_means_sha_before == class_means_sha_after
    )

    print(
        "H007B_DIAG "
        + json.dumps(
            {
                "stage": int(getattr(self, "_cur_task", -1)),
                "nb_old": nb_old,
                "nb_total": nb_total,
                "increment": nb_total - nb_old,
                "alignment_applied": bool(applied),
                "gamma": gamma,
                "fc_weight_norm_old_mean": old_norm_mean,
                "fc_weight_norm_new_mean_before": new_norm_mean,
                "fc_weight_norm_new_mean_after": aligned_new_norm_mean,
                "cnn_aligned_grouped": accuracy(aligned_preds.T[0], y_true, nb_old),
                "cnn_persistent_grouped_in_run_control": accuracy(
                    unaligned_preds.T[0], y_true, nb_old
                ),
                "aligned_equals_unaligned_topk": bool(np.array_equal(aligned_preds, unaligned_preds)),
                "fc_weight_sha256_before": fc_weight_sha_before,
                "fc_weight_sha256_after": fc_weight_sha_after,
                "fc_bias_sha256_before": fc_bias_sha_before,
                "fc_bias_sha256_after": fc_bias_sha_after,
                "network_state_sha256_before": state_sha_before,
                "network_state_sha256_after": state_sha_after,
                "class_means_sha256_before": class_means_sha_before,
                "class_means_sha256_after": class_means_sha_after,
                "persistent_state_unchanged": persistent_unchanged,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if not persistent_unchanged:
        raise RuntimeError(
            "H007b eval-only alignment mutated persistent state "
            "(fc.weight/fc.bias/state_dict/_class_means); experiment invalid"
        )

    return aligned_preds, y_true


def _h007b_eval_task(self, save_conf=False):
    """Observation-only wrapper: hash every persistent eval input/output state.

    ``eval_task`` is not overridden by iCaRL, so patching ``BaseLearner`` reaches
    it. This wrapper computes no prediction and changes no argument; it only
    proves that the whole evaluation (CNN eval + NME eval) leaves the network,
    the exemplar class means and the exemplar memory byte-identical.
    """
    net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

    def _snapshot():
        return {
            "network_state_sha256": _h007b_sha256_network(net),
            "class_means_sha256": _h007b_sha256_array(getattr(self, "_class_means", None)),
            "data_memory_sha256": _h007b_sha256_array(getattr(self, "_data_memory", None)),
            "targets_memory_sha256": _h007b_sha256_array(getattr(self, "_targets_memory", None)),
        }

    before = _snapshot()
    result = _H007B_ORIGINAL_EVAL_TASK(self, save_conf)
    after = _snapshot()
    unchanged = before == after

    print(
        "H007B_EVAL_STATE "
        + json.dumps(
            {
                "stage": int(getattr(self, "_cur_task", -1)),
                "before": before,
                "after": after,
                "persistent_state_unchanged": bool(unchanged),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if not unchanged:
        raise RuntimeError(
            "H007b evaluation changed persistent state (network/class_means/memory); "
            "experiment invalid"
        )

    return result


def _install_h007b_eval_only_alignment():
    """Install the eval-only alignment exactly once (no-op if already installed)."""
    global _H007B_ORIGINAL_EVAL_TASK
    if getattr(BaseLearner._eval_cnn, "_h007b_eval_only_alignment", False):
        return False
    _h007b_eval_cnn._h007b_eval_only_alignment = True
    _H007B_ORIGINAL_EVAL_TASK = BaseLearner.eval_task
    BaseLearner._eval_cnn = _h007b_eval_cnn
    BaseLearner.eval_task = _h007b_eval_task
    return True


_H007B_ORIGINAL_EVAL_TASK = BaseLearner.eval_task
_H007B_PATCH_INSTALLED = _install_h007b_eval_only_alignment()
print(
    "H007B_PATCH_INSTALLED "
    + json.dumps(
        {
            "installed": _H007B_PATCH_INSTALLED,
            "target": "BaseLearner._eval_cnn + observation-only BaseLearner.eval_task wrapper",
            "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
            "intervention": "eval-only new-class weight-norm alignment on a copied weight tensor",
            "persistent_state_mutation": False,
            "in_place_network_weight_align_called": False,
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
        "memory_size": 2000,
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
