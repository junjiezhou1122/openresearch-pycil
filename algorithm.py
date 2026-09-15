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
H017: is herding near-optimal, or can a coverage-oriented rule extract more?
-------------------------------------------------------------------------------
Configuration is the H013 stack: iCaRL + ResNet-18, init_epoch=100, epochs=50,
seed 1993, memory_size = 4000. Everything except the exemplar SELECTION rule is
identical to H013/H015/H016.

Where this comes from. Wave 3 established that the exemplar budget is the only
lever that has ever moved the accepted metric (H013: +2.7666 pp aggregate NME from
doubling memory_size 2000 -> 4000), that the gain is replay/representation-side
rather than prototype-side (H014), and that the stored set's INFORMATION CONTENT is
worth about 1 pp at that budget: herding beats uniform random selection by 1.000 pp
(H015) and that deficit replicated on an independent draw at 1.165 pp (H016). But
those three experiments only compared herding against RANDOM - the no-selection
floor. The open question they leave is whether herding is close to optimal for this
objective, or whether a coverage-oriented rule extracts still more from the same
fixed budget. That is exactly the question with a path to further utility at ZERO
extra memory.

Hypothesis. Herding selects exemplars so their running mean matches the class mean.
H017 instead selects a spread-out, coverage-oriented subset by greedy k-center
(farthest-first traversal) in the same L2-normalized feature space: start from the
sample nearest the class mean, then repeatedly add the sample farthest from the
already-selected set. If selection quality is a real, improvable axis, k-center
should beat herding; if herding is already near the ceiling for this metric,
k-center should not.

Falsifiable claim: greedy k-center selection beats iCaRL's herding selection by at
least +0.5 pp of aggregate NME at the same 4000-exemplar budget. The 0.5 pp margin
is chosen because the same-stack deterministic noise floor is 0.0 pp and the
observed spread between two independent random draws is only 0.165 pp, so +0.5 pp
is comfortably above both.

Minimal intervention: algorithm.py ONLY. memory_size stays 4000 (the H013 stack) and
the training loop, KD, optimiser, schedule, evaluator, split and seed are untouched.
The single change is that `BaseLearner._construct_exemplar` selects the ``m``
exemplars of each newly introduced class by greedy k-center instead of by herding;
the rest of that method (per-class feature extraction, the exemplar-mean
recomputation from the SELECTED exemplars, the memory append and the `_class_means`
update) is mirrored line for line from the frozen implementation. Old-class exemplars
still come from the upstream `_reduce_exemplar` truncation.

Determinism and RNG discipline: the k-center rule is fully deterministic with
index-order tie-breaking, so it draws NO randomness at all; the CPU self-check
verifies that the global torch RNG state after the call is IDENTICAL to the frozen
implementation's (the H009 failure mode cannot occur here).

Because this changes the stored training data, no exact invariance control is
available; the registered comparison is against the frozen H013 herding baseline
(aggregate 68.81833333), the H015/H016 random-selection draws (67.81833333 and
67.6533), and H001 (0.6605167), with the measured 0.0 pp same-stack noise floor.
"""

import hashlib
import json

import numpy as np

import models.base as base_module

_H017_PATCH_INSTALLED = False
_H017_SELECTION = "greedy_kcenter_farthest_first"
_H017_MARGIN_PP = 0.5
_H017_ROUND = 6


def _h017_sha_array(value):
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _h017_kcenter_indices(vectors, class_mean, m):
    """Deterministic greedy k-center (farthest-first) selection.

    Seed = the sample nearest the class mean; then repeatedly add the sample whose
    distance to the nearest already-selected sample is largest. Index-order
    tie-breaking keeps the rule fully deterministic.
    """
    n = len(vectors)
    take = int(min(m, n))
    if take <= 0:
        return np.zeros((0,), dtype=int)
    d_to_mean = np.sum((vectors - class_mean) ** 2, axis=1)
    first = int(np.argmin(d_to_mean))
    selected = [first]
    # min squared distance from every sample to the selected set
    best = np.sum((vectors - vectors[first]) ** 2, axis=1)
    best[first] = -1.0
    while len(selected) < take:
        nxt = int(np.argmax(best))
        selected.append(nxt)
        new_d = np.sum((vectors - vectors[nxt]) ** 2, axis=1)
        best = np.minimum(best, new_d)
        best[np.asarray(selected, dtype=int)] = -1.0
    return np.asarray(selected, dtype=int)


def _h017_construct_exemplar_kcenter(self, data_manager, m):
    """iCaRL's `_construct_exemplar` with the herding loop replaced by greedy k-center."""
    import logging

    logging.info(
        "[H017] Constructing exemplars by GREEDY K-CENTER selection...({} per class)".format(m)
    )
    DataLoader = base_module.DataLoader
    EPSILON = base_module.EPSILON
    batch_size = base_module.batch_size

    for class_idx in range(self._known_classes, self._total_classes):
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

        chosen = _h017_kcenter_indices(vectors, class_mean, m)
        selected_exemplars = np.array([np.array(data[i]) for i in chosen])
        exemplar_targets = np.full(len(chosen), class_idx)

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

        # Exemplar mean, exactly as the frozen implementation computes it.
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


def _install_h017_kcenter_selection():
    global _H017_PATCH_INSTALLED
    if _H017_PATCH_INSTALLED:
        return False

    base_module.BaseLearner._construct_exemplar = _h017_construct_exemplar_kcenter
    _H017_PATCH_INSTALLED = True

    print(
        "H017_PATCH_INSTALLED "
        + json.dumps(
            {
                "target": "BaseLearner._construct_exemplar",
                "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
                "memory_size": 4000,
                "selection": _H017_SELECTION,
                "deterministic_no_rng": True,
                "margin_pp": _H017_MARGIN_PP,
                "old_class_selection": "upstream _reduce_exemplar truncation unchanged",
                "training_loop_untouched": True,
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
    # H017: the H013 stack (memory 4000) with new-class exemplar selection changed
    # from herding to deterministic greedy k-center, at the SAME budget, to test
    # whether the ~1 pp selection-content term can be enlarged.
    _install_h017_kcenter_selection()
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
