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
H023: is the surviving headroom an ALLOCATION artefact? (age-weighted budget)
-------------------------------------------------------------------------------
Configuration is the H018 stack: iCaRL + ResNet-18, init_epoch=100, epochs=50,
seed 1993 (harness-pinned), memory_size = 6000. The single change is HOW the budget
is spread across classes.

Where this comes from. Waves 3-5 characterised the exemplar budget as the only lever
that has ever moved the accepted metric, and then exhausted it in two of its three
formulations:
  * total budget: 2000 -> 4000 -> 6000 gives aggregate NME 66.0517 -> 68.8183 ->
    70.8217 (+4.77 pp cumulative), and H021 showed the full-train oracle now beats
    the registered rule by only +0.5433 pp, i.e. ~77% of the full-data headroom is
    already consumed - so buying more memory is no longer productive.
  * selection rule: herding beats uniform random (H015/H016, replicated) but greedy
    k-center is 3.00 pp WORSE than herding and below both random draws (H017), so
    selection is at its ceiling.
  * allocation: NOT YET TESTED. iCaRL spreads the budget uniformly as
    samples_per_class = memory_size // total_classes, so the oldest classes keep the
    same per-class allowance as the newest - even though H021 located the residual
    headroom precisely at LATE-STAGE OLD-CLASS retention (+1.96/+1.40 pp remaining
    at stages 4-5).

Hypothesis. The surviving headroom is partly an allocation artefact: giving older
classes more exemplars and the newest classes fewer, at the SAME total budget, will
raise aggregate NME above H018's 70.8217.

Registered allocation rule: with total classes and base allowance m = memory_size //
total_classes, class c (0 = oldest) receives
    allow(c) = max(1, round(m * (1 + alpha * (1 - 2*(c + 0.5)/total))))
with alpha = 0.5 registered a priori. The weights sum to exactly `total`, so the
total budget is preserved: the oldest classes get up to 1.5x the uniform allowance
and the newest get down to 0.5x. alpha is a single a-priori constant, never tuned.

Pre-registered reading: SUPPORTED if aggregate NME exceeds H018's 70.8217 by at
least +0.5 pp; REFUTED if it does not, in which case the memory axis is exhausted in
all three formulations (curve flat near the frontier, selection at its ceiling,
allocation inert).

Minimal intervention: algorithm.py ONLY. `BaseLearner._reduce_exemplar` and
`BaseLearner._construct_exemplar` are replaced by versions that mirror the frozen
implementations line for line - same per-class feature extraction, same herding loop,
same exemplar-mean recomputation, same memory append - except that the per-class
allowance is `allow(class_idx)` instead of the scalar `m`. A per-stage
`H023_ALLOC` marker prints the allowance summary so the allocation change is provably
active in the run's own stdout (standing rule 7).

No exact invariance control is available: the intervention changes which exemplars
are stored, hence the training data and (through the changed number of DataLoader
iterations) the global RNG stream, exactly as the accepted H013/H015/H018
data-changing interventions did. The registered comparison is against H018's 6000
budget run (aggregate 70.8217) with the seed-pinned 0.0 pp same-stack noise floor.
"""

import hashlib
import json

import numpy as np

import models.base as base_module

_H023_PATCH_INSTALLED = False
_H023_ALPHA = 0.5
_H023_MARGIN_PP = 0.5
_H023_LAST_ALLOWANCES = None


def _h023_allowances(self, m):
    """Registered age-weighted per-class exemplar allowances (sum ~= total * m)."""
    total = int(self._total_classes)
    if total <= 0:
        return []
    out = []
    for c in range(total):
        w = 1.0 + _H023_ALPHA * (1.0 - 2.0 * (c + 0.5) / total)
        out.append(int(max(1, round(m * w))))
    return out


def _h023_log_allocations(self, phase):
    global _H023_LAST_ALLOWANCES
    allow = _H023_LAST_ALLOWANCES or []
    if not allow:
        return
    print(
        "H023_ALLOC "
        + json.dumps(
            {
                "phase": phase,
                "stage": int(getattr(self, "_cur_task", -1)),
                "total_classes": int(self._total_classes),
                "known_classes": int(self._known_classes),
                "alpha": _H023_ALPHA,
                "allowance_sum": int(sum(allow)),
                "allowance_min": int(min(allow)),
                "allowance_max": int(max(allow)),
                "oldest_allowance": int(allow[0]),
                "newest_allowance": int(allow[-1]),
                "uniform_allowance": int(self.samples_per_class),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _h023_reduce_exemplar(self, data_manager, m):
    """Frozen `_reduce_exemplar`, with a per-class (age-weighted) allowance."""
    import copy
    import logging

    DataLoader = base_module.DataLoader
    EPSILON = base_module.EPSILON
    batch_size = base_module.batch_size

    logging.info("Reducing exemplars...(age-weighted, base {} per class)".format(m))
    allow = _h023_allowances(self, m)
    global _H023_LAST_ALLOWANCES
    _H023_LAST_ALLOWANCES = allow

    dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
        self._targets_memory
    )
    self._class_means = np.zeros((self._total_classes, self.feature_dim))
    self._data_memory, self._targets_memory = np.array([]), np.array([])

    for class_idx in range(self._known_classes):
        k = allow[class_idx] if class_idx < len(allow) else int(m)
        mask = np.where(dummy_targets == class_idx)[0]
        dd, dt = dummy_data[mask][:k], dummy_targets[mask][:k]
        self._data_memory = (
            np.concatenate((self._data_memory, dd))
            if len(self._data_memory) != 0
            else dd
        )
        self._targets_memory = (
            np.concatenate((self._targets_memory, dt))
            if len(self._targets_memory) != 0
            else dt
        )

        idx_dataset = data_manager.get_dataset(
            [], source="train", mode="test", appendent=(dd, dt)
        )
        idx_loader = DataLoader(
            idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        vectors, _ = self._extract_vectors(idx_loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        mean = np.mean(vectors, axis=0)
        mean = mean / np.linalg.norm(mean)
        self._class_means[class_idx, :] = mean

    _h023_log_allocations(self, "reduce")


def _h023_construct_exemplar(self, data_manager, m):
    """Frozen `_construct_exemplar`, with a per-class (age-weighted) allowance."""
    import logging

    DataLoader = base_module.DataLoader
    EPSILON = base_module.EPSILON
    batch_size = base_module.batch_size

    logging.info("Constructing exemplars...(age-weighted, base {} per class)".format(m))
    global _H023_LAST_ALLOWANCES
    allow = _H023_LAST_ALLOWANCES
    if not allow:
        allow = _h023_allowances(self, m)
        _H023_LAST_ALLOWANCES = allow

    for class_idx in range(self._known_classes, self._total_classes):
        k_allow = allow[class_idx] if class_idx < len(allow) else int(m)
        data, targets, idx_dataset = data_manager.get_dataset(
            np.arange(class_idx, class_idx + 1),
            source="train",
            mode="test",
            ret_data=True,
        )
        idx_loader = DataLoader(
            idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        vectors, _ = self._extract_vectors(idx_loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        class_mean = np.mean(vectors, axis=0)

        m_eff = int(min(k_allow, len(vectors)))
        selected_exemplars = []
        exemplar_vectors = []
        for k in range(1, m_eff + 1):
            S = np.sum(exemplar_vectors, axis=0)
            mu_p = (vectors + S) / k
            i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
            selected_exemplars.append(np.array(data[i]))
            exemplar_vectors.append(np.array(vectors[i]))
            vectors = np.delete(vectors, i, axis=0)
            data = np.delete(data, i, axis=0)

        selected_exemplars = np.array(selected_exemplars)
        exemplar_targets = np.full(m_eff, class_idx)
        self._data_memory = (
            np.concatenate((self._data_memory, selected_exemplars))
            if len(self._data_memory) != 0
            else selected_exemplars
        )
        self._targets_memory = (
            np.concatenate((self._targets_memory, exemplar_targets))
            if len(self._targets_memory) != 0
            else exemplar_targets
        )

        idx_dataset = data_manager.get_dataset(
            [], source="train", mode="test", appendent=(selected_exemplars, exemplar_targets)
        )
        idx_loader = DataLoader(
            idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        vectors, _ = self._extract_vectors(idx_loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        mean = np.mean(vectors, axis=0)
        mean = mean / np.linalg.norm(mean)
        self._class_means[class_idx, :] = mean

    _h023_log_allocations(self, "construct")


def _install_h023_age_weighted_allocation():
    global _H023_PATCH_INSTALLED
    if _H023_PATCH_INSTALLED:
        return False

    base_module.BaseLearner._reduce_exemplar = _h023_reduce_exemplar
    base_module.BaseLearner._construct_exemplar = _h023_construct_exemplar
    _H023_PATCH_INSTALLED = True

    print(
        "H023_PATCH_INSTALLED "
        + json.dumps(
            {
                "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
                "memory_size": 6000,
                "alpha": _H023_ALPHA,
                "margin_pp": _H023_MARGIN_PP,
                "targets": [
                    "BaseLearner._reduce_exemplar",
                    "BaseLearner._construct_exemplar",
                ],
                "effect": "age-weighted per-class allowances at a constant total budget",
                "proof_of_effect": "one H023_ALLOC marker per exemplar phase per stage",
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
    # H023: the H018 stack (memory 6000) with the exemplar budget allocated
    # age-weighted across classes instead of uniformly.
    _install_h023_age_weighted_allocation()
    return {
        "prefix": "benchmark",
        "dataset": "cifar100",
        "memory_size": 6000,   # H018 stack
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
