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
H016: does the ~1 pp herding advantage REPLICATE on an independent draw?
-------------------------------------------------------------------------------
Configuration, intervention and controls are identical to H015 (wave 3): the H013
stack (memory_size 4000) with `BaseLearner._construct_exemplar` selecting the newly
introduced classes' exemplars UNIFORMLY AT RANDOM instead of by herding, everything
else mirrored line for line.

The ONE substantive difference is the random draw: H015 used a local
`np.random.RandomState(1993 + class_idx)`; H016 uses `7919 + class_idx`, an
independent draw from the same rule.

Why this is the right next experiment. H015 (valid/supported) found that herding
beats random selection by exactly 1.0000 pp of aggregate NME (68.8183 vs 67.8183)
at the same budget - a value sitting exactly ON its pre-registered >=1.0 pp margin,
on a single seed. Its verifier recorded that as its first bounding caveat: any
sub-0.01 pp drift would flip the label. Because the rest of the stack is
deterministic (identical trees reproduce to 0.0 pp reported precision), the only
source of variation that could matter is the random draw itself. Re-running the
SAME rule with a DIFFERENT seed therefore tests the claim that matters - is the
~1 pp content effect a property of the SELECTION RULE, or of one fortunate draw?

Falsifiable claim: the herding advantage is reproducible, i.e. a second independent
random selection also loses at least 1.0 pp of aggregate NME to herding. If the
second draw lands within 1.0 pp of herding (or above it), the H015 margin result is
draw-specific and the ~1 pp "content" term is inside selection variance.

RNG discipline is unchanged: the draw uses a LOCAL RandomState, so no randomness
beyond what the frozen implementation also draws is taken from the global
torch/numpy RNG, and the CPU self-check verifies that the global torch RNG state
after the call is IDENTICAL to the frozen implementation's.
"""

import hashlib
import json

import numpy as np

import models.base as base_module

_H016_PATCH_INSTALLED = False
_H016_SELECTION = "random_uniform_per_class"
_H016_RNG_SEED = 7919


def _h016_sha_array(value):
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _h016_construct_exemplar_random(self, data_manager, m):
    """iCaRL's `_construct_exemplar` with the herding loop replaced by random choice.

    Everything else - the feature extraction, the exemplar-mean recomputation from
    the selected exemplars, the memory append and the `_class_means` update - is
    mirrored from the frozen implementation.
    """
    import logging

    logging.info(
        "[H016] Constructing exemplars by RANDOM selection...({} per class)".format(m)
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
        local_rng = np.random.RandomState(_H016_RNG_SEED + int(class_idx))
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


def _install_h016_random_selection():
    global _H016_PATCH_INSTALLED
    if _H016_PATCH_INSTALLED:
        return False

    base_module.BaseLearner._construct_exemplar = _h016_construct_exemplar_random
    _H016_PATCH_INSTALLED = True

    print(
        "H016_PATCH_INSTALLED "
        + json.dumps(
            {
                "target": "BaseLearner._construct_exemplar",
                "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
                "memory_size": 4000,
                "selection": _H016_SELECTION,
                "local_rng_seed": _H016_RNG_SEED,
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
    # H016: the H013 stack (memory 4000) with the new-class exemplar selection
    # changed from herding to uniform random, at the SAME budget, to test whether
    # the budget's information content matters or only its volume.
    _install_h016_random_selection()
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
