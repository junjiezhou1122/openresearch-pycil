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
H015: does the exemplar budget's INFORMATION CONTENT matter, or only its VOLUME?
-------------------------------------------------------------------------------
Configuration is the H013 stack: iCaRL + ResNet-18, init_epoch=100, epochs=50,
seed 1993, memory_size = 4000. Everything except the exemplar SELECTION rule is
identical to H013.

Where this comes from. H013 (valid/supported) doubled the budget and moved the
metric of record for the first time (aggregate NME 0.6605167 -> 0.6881833,
+2.7666 pp). H014 (valid/refuted) then decomposed that gain: rolling the NME
prototype table back to H001's per-class budget on the same representation costs
only 0.0483 pp, so the gain is carried by the TRAINING/replay channel (more
replay data in the KD and classifier losses, CNN aggregate +5.31 pp), not by
better prototypes.

That makes the next question sharp. If the gain is about how much INFORMATION the
replay set carries, then replacing iCaRL's herding selection with a RANDOM
selection of the SAME number of exemplars per class should degrade it noticeably.
If it is merely about replay VOLUME, random selection should be nearly as good.
Herding exists precisely to make the exemplar mean approximate the class mean, so
this isolates "informativeness of the stored set" from "number of stored samples".

Minimal intervention: algorithm.py ONLY. memory_size stays 4000 (the H013 stack)
and the base stage, training loop, KD, optimiser, schedule, evaluator, split and
seed are untouched. The single change is that `BaseLearner._construct_exemplar`
selects the ``m`` exemplars of each newly introduced class UNIFORMLY AT RANDOM
instead of by herding; the rest of that method (feature extraction, the class-mean
computation from the selected exemplars, the memory append and `_class_means`
update) is mirrored line for line from the frozen implementation. Old-class
exemplars are still produced by the upstream `_reduce_exemplar` truncation, so the
contrast is confined to the new classes' selection - the part herding actually
governs.

RNG discipline: the random draw uses a LOCAL ``np.random.RandomState(1993 +
class_idx)``, so no draw is taken from the global torch/numpy RNG and the training
stream is not perturbed beyond the intended change (the H009 failure mode is
avoided by construction).

Because this changes the stored training data, no exact invariance control is
available; the registered comparison is against the frozen H013 baseline
(68.8183) and H001 (0.6605167) with the measured 0.0 pp same-stack noise floor.
"""

import hashlib
import json

import numpy as np

import models.base as base_module

_H015_PATCH_INSTALLED = False
_H015_SELECTION = "random_uniform_per_class"
_H015_RNG_SEED = 1993


def _h015_sha_array(value):
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _h015_construct_exemplar_random(self, data_manager, m):
    """iCaRL's `_construct_exemplar` with the herding loop replaced by random choice.

    Everything else - the feature extraction, the exemplar-mean recomputation from
    the selected exemplars, the memory append and the `_class_means` update - is
    mirrored from the frozen implementation.
    """
    import logging

    logging.info(
        "[H015] Constructing exemplars by RANDOM selection...({} per class)".format(m)
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

        n_available = len(data)
        n_take = int(min(m, n_available))
        # Local RNG only: never touches the global torch/numpy stream.
        local_rng = np.random.RandomState(_H015_RNG_SEED + int(class_idx))
        chosen = local_rng.choice(n_available, size=n_take, replace=False)

        selected_exemplars = np.array([np.array(data[i]) for i in chosen])
        exemplar_targets = np.full(n_take, class_idx)

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


def _install_h015_random_selection():
    global _H015_PATCH_INSTALLED
    if _H015_PATCH_INSTALLED:
        return False

    base_module.BaseLearner._construct_exemplar = _h015_construct_exemplar_random
    _H015_PATCH_INSTALLED = True

    print(
        "H015_PATCH_INSTALLED "
        + json.dumps(
            {
                "target": "BaseLearner._construct_exemplar",
                "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
                "memory_size": 4000,
                "selection": _H015_SELECTION,
                "local_rng_seed": _H015_RNG_SEED,
                "global_rng_untouched": True,
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
    # H015: the H013 stack (memory 4000) with the new-class exemplar selection
    # changed from herding to uniform random, at the SAME budget, to test whether
    # the budget's information content matters or only its volume.
    _install_h015_random_selection()
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
