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
H009c: EVALUATION-ONLY full-data new-class NME prototypes (readout-isolated)
-------------------------------------------------------------------------------
Everything about training is identical to the verified H001 configuration
(iCaRL + ResNet-18, init_epoch=100, epochs=50, seed 1993, memory_size=2000).

The single change is an *evaluation-time* monkeypatch of
``BaseLearner._eval_nme`` (inherited by ``iCaRL``). When a stage is scored, a
**copy** of ``self._class_means`` is built whose rows for the classes introduced
at this stage (``[self._known_classes, self._total_classes)``) are replaced by
the full-data normalized prototype of that class (extract features for every
train sample of the class, L2-normalize each feature, mean, re-normalize --
exactly the existing ``L2-normalize -> mean -> L2-normalize`` procedure). The
nearest-mean prediction is then made against that copy.

Why this design (H009b repair): the invalid H009b run recomputed the prototype
table on the *persistent* object inside ``_construct_exemplar`` and registered
"old-class NME exact invariance" as a control. That control is not a true
invariance control, because NME is an argmin over ALL class means, so recomputed
new-class prototypes necessarily compete for old-class test samples (H009b
measured -0.05..-0.09 pp old-class NME drift). Here the persistent
``_class_means`` is never written, so the true control is exact by
construction, and the effect is scored with an old-prototype-restricted argmin
(which cannot be perturbed by new-class prototypes) plus sha256 byte-equality of
every persistent object.

Scope / auditability:
  * ``self._class_means``, ``_data_memory``, ``_targets_memory``, ``fc.weight``,
    ``fc.bias``, the whole ``state_dict`` and the convnet are never written;
    sha256 checksums are compared before/after every ``_eval_nme`` call, printed
    per stage, and any mismatch raises (fail fast).
  * Building/iterating any DataLoader draws ``_base_seed`` from the *global*
    torch generator, so the extra feature-extraction loaders are wrapped in a
    global CPU **and CUDA** RNG snapshot/restore, and both restorations are
    verified and printed per stage (the H009b verifier asked for the CUDA check).
  * The unaltered persistent head is scored in the SAME forward pass and emitted
    as an in-run control, so the run proves the eval-time copy never leaks.
  * The returned prediction is the patched (full-data prototype) one.
"""

import hashlib
import json

import numpy as np
import torch

import models.base as base_module
from scipy.spatial.distance import cdist

_H009C_PATCH_INSTALLED = False
_H009C_EPS = base_module.EPSILON


def _h009c_sha_array(value):
    """Content hash of a numpy-like array (None-safe)."""
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _h009c_sha_tensor(tensor):
    """Content hash of one tensor (device-independent)."""
    return hashlib.sha256(
        tensor.detach().to("cpu").contiguous().numpy().tobytes()
    ).hexdigest()


def _h009c_sha_network(net):
    """Content hash of a whole network state_dict."""
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(_h009c_sha_tensor(state[name]).encode("ascii"))
    return digest.hexdigest()


def _h009c_jsonable_grouped(grouped):
    """numpy-float-safe copy of an accuracy() grouped dict."""
    return {str(k): (float(v) if v is not None else None) for k, v in grouped.items()}


def _h009c_eval_nme(self, loader, class_means):
    """H009c: eval-only NME with full-data prototypes for the newest classes.

    Returns ``(y_pred, y_true)`` with the same contract as
    ``BaseLearner._eval_nme`` ([N, topk] int array, [N] int array) so
    ``eval_task`` / ``_evaluate`` consume it unchanged. The returned prediction
    comes from the *copied, partially recomputed* prototype table; the
    persistent table is only read and is proven unchanged by before/after hashes.
    """
    import torch.nn as nn

    data_manager = getattr(self, "_h009c_data_manager", None)
    nb_old = int(self._known_classes)
    nb_total = int(self._total_classes)
    topk = int(self.topk)

    net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

    persistent = class_means
    cm_before = _h009c_sha_array(persistent)
    data_before = _h009c_sha_array(self._data_memory)
    targets_before = _h009c_sha_array(self._targets_memory)
    weight_before = _h009c_sha_tensor(net.fc.weight)
    bias_before = None if net.fc.bias is None else _h009c_sha_tensor(net.fc.bias)
    network_before = _h009c_sha_network(self._network)

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    patched = np.array(persistent, dtype=persistent.dtype, copy=True)
    applied = False
    new_proto_shift = []
    try:
        if data_manager is not None and nb_total > nb_old:
            for class_idx in range(nb_old, nb_total):
                _, _, class_dset = data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                    ret_data=True,
                )
                class_loader = base_module.DataLoader(
                    class_dset,
                    batch_size=base_module.batch_size,
                    shuffle=False,
                    num_workers=4,
                )
                vectors, _ = self._extract_vectors(class_loader)
                vectors = (
                    vectors.T / (np.linalg.norm(vectors.T, axis=0) + _H009C_EPS)
                ).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)
                new_proto_shift.append(
                    float(np.linalg.norm(mean - persistent[class_idx, :]))
                )
                patched[class_idx, :] = mean
            applied = True
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    rng_cpu_restored = bool(torch.get_rng_state().equal(cpu_rng))
    rng_cuda_restored = None
    if cuda_rng is not None:
        current_cuda = torch.cuda.get_rng_state_all()
        rng_cuda_restored = bool(
            len(current_cuda) == len(cuda_rng)
            and all(a.equal(b) for a, b in zip(current_cuda, cuda_rng))
        )

    # ---- score both heads on the same extracted test features ----
    self._network.eval()
    vectors_t, y_true = self._extract_vectors(loader)
    vectors_t = (vectors_t.T / (np.linalg.norm(vectors_t.T, axis=0) + _H009C_EPS)).T

    scores_persistent = cdist(persistent, vectors_t, "sqeuclidean").T
    preds_persistent = np.argsort(scores_persistent, axis=1)[:, :topk]
    scores_patched = cdist(patched, vectors_t, "sqeuclidean").T
    preds_patched = np.argsort(scores_patched, axis=1)[:, :topk]

    # ---- true invariance control: argmin restricted to OLD prototypes ----
    def _old_restricted_acc(means, targets, nb):
        if nb <= 0:
            return None
        mask = targets < nb
        if not mask.any():
            return None
        dists = cdist(means[:nb], vectors_t[mask], "sqeuclidean")
        preds = np.argmin(dists, axis=0)
        return float(np.mean(preds == targets[mask]) * 100)

    old_restricted_persistent = _old_restricted_acc(persistent, y_true, nb_old)
    old_restricted_patched = _old_restricted_acc(patched, y_true, nb_old)

    cm_after = _h009c_sha_array(persistent)
    data_after = _h009c_sha_array(self._data_memory)
    targets_after = _h009c_sha_array(self._targets_memory)
    weight_after = _h009c_sha_tensor(net.fc.weight)
    bias_after = None if net.fc.bias is None else _h009c_sha_tensor(net.fc.bias)
    network_after = _h009c_sha_network(self._network)

    flags = {
        "class_means_sha_unchanged": cm_after == cm_before,
        "data_memory_sha_unchanged": data_after == data_before,
        "targets_memory_sha_unchanged": targets_after == targets_before,
        "fc_weight_sha_unchanged": weight_after == weight_before,
        "fc_bias_sha_unchanged": bias_after == bias_before,
        "network_sha_unchanged": network_after == network_before,
        "rng_cpu_restored": rng_cpu_restored,
        "rng_cuda_restored": rng_cuda_restored,
    }
    invariance_ok = all(
        v for k, v in flags.items() if v is not None
    )

    diag = {
        "stage": int(self._cur_task),
        "nb_old": nb_old,
        "nb_total": nb_total,
        "recompute_applied": applied,
        "recomputed_classes": int(max(nb_total - nb_old, 0)) if applied else 0,
        "new_prototype_shift_mean": (
            float(np.mean(new_proto_shift)) if new_proto_shift else None
        ),
        "new_prototype_shift_max": (
            float(np.max(new_proto_shift)) if new_proto_shift else None
        ),
        "nme_persistent_grouped_in_run_control": _h009c_jsonable_grouped(
            base_module.accuracy(preds_persistent.T[0], y_true, nb_old)
        ),
        "nme_patched_grouped": _h009c_jsonable_grouped(
            base_module.accuracy(preds_patched.T[0], y_true, nb_old)
        ),
        "old_restricted_acc_persistent": old_restricted_persistent,
        "old_restricted_acc_patched": old_restricted_patched,
        "invariance_flags": flags,
        "invariance_ok": bool(invariance_ok),
    }
    print("H009C_DIAG " + json.dumps(diag, sort_keys=True), flush=True)

    if not invariance_ok:
        raise RuntimeError(
            "H009C invariance violation (persistent state or RNG changed): "
            + json.dumps(diag, sort_keys=True)
        )

    return preds_patched, y_true


def _install_h009c_eval_prototypes():
    """Install the eval-only full-data prototype patch exactly once."""
    global _H009C_PATCH_INSTALLED
    if _H009C_PATCH_INSTALLED:
        return False

    from models.icarl import iCaRL

    original_incremental_train = iCaRL.incremental_train

    def _h009c_incremental_train(self, data_manager):
        # Keep a handle on the data manager so _eval_nme can reach the full
        # train split. This stores no numeric state and writes nothing.
        self._h009c_data_manager = data_manager
        return original_incremental_train(self, data_manager)

    iCaRL.incremental_train = _h009c_incremental_train
    base_module.BaseLearner._eval_nme = _h009c_eval_nme
    _H009C_PATCH_INSTALLED = True

    print(
        "H009C_PATCH_INSTALLED "
        + json.dumps(
            {
                "target": "BaseLearner._eval_nme (inherited by iCaRL)",
                "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
                "mode": "evaluation-only; persistent _class_means never written",
                "rng_guard": "cpu+cuda snapshot/restore with verification",
                "training_untouched": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return True


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
    # H009c: evaluation-only recomputation of the newest classes' NME prototypes
    # from all available training samples. Training, memory budget, exemplar
    # selection, the persistent prototype table and the RNG stream are unchanged.
    _install_h009c_eval_prototypes()
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
