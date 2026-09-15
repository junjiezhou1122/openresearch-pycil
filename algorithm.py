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
H014: where did the H013 gain come from - the prototypes or the representation?
-------------------------------------------------------------------------------
Configuration is the H013 stack: iCaRL + ResNet-18, init_epoch=100, epochs=50,
seed 1993, and memory_size = 4000 (the doubled budget). Nothing about training,
memory, exemplar selection or the evaluator differs from H013.

Why this experiment. H013 (valid/supported, wave 3) doubled the exemplar budget
2000 -> 4000 and moved the metric of record for the first time:
aggregate NME 0.6605167 -> 0.6881833 (+2.7666 pp), entirely as OLD-class retention
(+3.13..+5.33 pp at every incremental stage). But the budget change is
STRUCTURALLY COUPLED: a larger exemplar set both (a) feeds more replay data into
KD and the classifier loss, improving the learned REPRESENTATION (H013's CNN
aggregate also rose 59.28 -> 64.59, +5.31 pp), and (b) supplies more samples for
the NME PROTOTYPE table. H013 cannot say which of the two carries the gain.

H014 decomposes it with an evaluation-only intervention on the SAME run. iCaRL's
herding is greedy and `_reduce_exemplar` truncates with `[:m]`, so the
2000-budget exemplar set is EXACTLY the prefix of the 4000-budget set. H014
therefore builds, at evaluation time only, a second prototype table using only
the first ``m_h001 = 2000 // total_classes`` exemplars per class - i.e. the
prototype information a 2000-exemplar learner would have - while keeping the
H013 representation.

  * If the reduced-prototype NME falls well below the full-budget NME, the H013
    gain is carried by the ENLARGED PROTOTYPE INFORMATION (prototype channel).
  * If it stays at the full-budget level, the gain is carried by the IMPROVED
    REPRESENTATION and the prototype table is incidental.

Scope / auditability:
  * Evaluation-only. The persistent `_class_means`, `_data_memory`,
    `_targets_memory`, `fc.weight`, `fc.bias` and the whole `state_dict` are never
    written; sha256 equality is checked before/after every `_eval_nme` call and a
    mismatch raises.
  * The added feature-extraction DataLoader is wrapped in a CPU **and** CUDA
    global torch RNG snapshot/restore whose restoration is verified per stage.
  * The FULL-budget registered NME is computed in the SAME pass and is the
    RETURNED prediction, so the run's official metric must reproduce H013's
    aggregate (68.8183) - a strong in-run control that the patch changed nothing
    but added a diagnostic.
  * The reduced-budget prototype accuracy is emitted on a machine-readable
    `H014_DIAG` line per stage.
"""

import hashlib
import json

import numpy as np
import torch

import models.base as base_module

_H014_PATCH_INSTALLED = False
_H014_EPS = base_module.EPSILON
_H014_H001_BUDGET = 2000


def _h014_sha_array(value):
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _h014_sha_tensor(tensor):
    return hashlib.sha256(
        tensor.detach().to("cpu").contiguous().numpy().tobytes()
    ).hexdigest()


def _h014_sha_network(net):
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(_h014_sha_tensor(state[name]).encode("ascii"))
    return digest.hexdigest()


def _h014_grouped(grouped):
    return {str(k): (float(v) if v is not None else None) for k, v in grouped.items()}


def _h014_normalize(x):
    return (x.T / (np.linalg.norm(x.T, axis=0) + _H014_EPS)).T


def _h014_eval_nme(self, loader, class_means):
    """H014: full-budget registered NME (control, returned) + reduced-budget diagnostic."""
    import torch.nn as nn

    from scipy.spatial.distance import cdist

    data_manager = getattr(self, "_h014_data_manager", None)
    nb_old = int(self._known_classes)
    nb_total = int(self._total_classes)
    topk = int(self.topk)

    net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

    persistent = class_means
    cm_before = _h014_sha_array(persistent)
    data_before = _h014_sha_array(self._data_memory)
    targets_before = _h014_sha_array(self._targets_memory)
    weight_before = _h014_sha_tensor(net.fc.weight)
    bias_before = None if net.fc.bias is None else _h014_sha_tensor(net.fc.bias)
    network_before = _h014_sha_network(self._network)

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    # The prototype information a 2000-exemplar learner would have: the first
    # m_h001 exemplars of each class (herding is greedy and _reduce_exemplar
    # truncates with [:m], so this is exactly the 2000-budget set).
    m_h001 = max(_H014_H001_BUDGET // max(nb_total, 1), 1)
    sub_x = sub_y = None
    n_sub = 0
    try:
        if data_manager is not None and len(self._targets_memory) != 0:
            keep = []
            targets_mem = np.asarray(self._targets_memory)
            for class_idx in range(nb_total):
                idx = np.where(targets_mem == class_idx)[0][:m_h001]
                keep.append(idx)
            keep = np.concatenate(keep) if keep else np.zeros((0,), dtype=int)
            sub_x = np.asarray(self._data_memory)[keep]
            sub_y = targets_mem[keep]
            n_sub = int(len(sub_y))
            sub_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(sub_x, sub_y)
            )
            sub_loader = base_module.DataLoader(
                sub_dset, batch_size=base_module.batch_size, shuffle=False, num_workers=4
            )
            sub_feat, sub_lab = self._extract_vectors(sub_loader)
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

    self._network.eval()
    vectors_t, y_true = self._extract_vectors(loader)
    vectors_t = _h014_normalize(vectors_t)

    # (a) registered full-budget NME - the returned prediction (in-run control)
    preds_full = np.argsort(cdist(persistent, vectors_t, "sqeuclidean").T, axis=1)[:, :topk]

    # (b) reduced-budget prototype table, built from the prefix subset features
    reduced_ok = False
    preds_reduced = preds_full
    n_classes_reduced = 0
    if sub_x is not None and n_sub > 0:
        feats = _h014_normalize(sub_feat)
        reduced_means = np.array(persistent, copy=True)
        for class_idx in range(nb_total):
            rows = feats[sub_lab == class_idx]
            if len(rows) == 0:
                continue
            m = rows.mean(axis=0)
            norm = np.linalg.norm(m)
            reduced_means[class_idx, :] = m / norm if norm > 0 else m
            n_classes_reduced += 1
        preds_reduced = np.argsort(
            cdist(reduced_means, vectors_t, "sqeuclidean").T, axis=1
        )[:, :topk]
        reduced_ok = True

    cm_after = _h014_sha_array(persistent)
    data_after = _h014_sha_array(self._data_memory)
    targets_after = _h014_sha_array(self._targets_memory)
    weight_after = _h014_sha_tensor(net.fc.weight)
    bias_after = None if net.fc.bias is None else _h014_sha_tensor(net.fc.bias)
    network_after = _h014_sha_network(self._network)

    flags = {
        "class_means_sha_unchanged": cm_after == cm_before,
        "data_memory_sha_unchanged": data_after == data_before,
        "targets_memory_sha_unchanged": targets_after == targets_before,
        "fc_weight_sha_unchanged": weight_after == weight_before,
        "fc_bias_sha_unchanged": bias_after == bias_before,
        "network_sha_unchanged": network_after == network_before,
        "rng_cpu_restored": rng_cpu_restored,
        "rng_cuda_restored": rng_cuda_restored,
        "reduced_table_built": reduced_ok,
    }
    invariance_ok = all(v for v in flags.values() if v is not None)

    diag = {
        "stage": int(self._cur_task),
        "nb_old": nb_old,
        "nb_total": nb_total,
        "h001_budget_per_class": m_h001,
        "n_reduced_subset_exemplars": n_sub,
        "n_classes_in_reduced_table": n_classes_reduced,
        "full_budget_grouped_in_run_control": _h014_grouped(
            base_module.accuracy(preds_full.T[0], y_true, nb_old)
        ),
        "reduced_budget_grouped": _h014_grouped(
            base_module.accuracy(preds_reduced.T[0], y_true, nb_old)
        ),
        "invariance_flags": flags,
        "invariance_ok": bool(invariance_ok),
    }
    print("H014_DIAG " + json.dumps(diag, sort_keys=True), flush=True)

    if not invariance_ok:
        raise RuntimeError(
            "H014 invariance violation (persistent state or RNG changed): "
            + json.dumps(diag, sort_keys=True)
        )

    # Return the registered FULL-budget prediction so the official metric is H013's.
    return preds_full, y_true


def _install_h014_decomposition():
    global _H014_PATCH_INSTALLED
    if _H014_PATCH_INSTALLED:
        return False

    from models.icarl import iCaRL

    original_incremental_train = iCaRL.incremental_train

    def _h014_incremental_train(self, data_manager):
        self._h014_data_manager = data_manager
        return original_incremental_train(self, data_manager)

    iCaRL.incremental_train = _h014_incremental_train
    base_module.BaseLearner._eval_nme = _h014_eval_nme
    _H014_PATCH_INSTALLED = True

    print(
        "H014_PATCH_INSTALLED "
        + json.dumps(
            {
                "target": "BaseLearner._eval_nme (inherited by iCaRL)",
                "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
                "mode": "evaluation-only decomposition; persistent state never written",
                "returned": "registered FULL-budget NME (H013 stack)",
                "diagnostic": "reduced-budget (2000-prefix) prototype table NME",
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
    - model_name: Switch to a different continual learning algorithm (der, foster, memo, etc.)
    - init_epoch / epochs: Number of training epochs per stage
    - memory_size: Total exemplar memory budget
    - init_cls / increment: Class-incremental schedule
    - convnet_type: Backbone architecture
    - Other hyperparameters specific to the chosen model
    """
    # H014: the H013 stack (memory 4000) plus an evaluation-only decomposition of
    # where the H013 gain came from - enlarged prototype information or the
    # concurrently improved representation.
    _install_h014_decomposition()
    return {
        "prefix": "benchmark",
        "dataset": "cifar100",
        "memory_size": 4000,   # H013 stack: doubled exemplar budget
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
