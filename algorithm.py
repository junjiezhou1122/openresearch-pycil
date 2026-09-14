"""
Baseline continual learning algorithm: iCaRL on CIFAR-100.

This file contains the epoch and hyperparameter configuration for iCaRL.
The actual iCaRL implementation is in the PyCIL repository (models/icarl.py).

Agents should modify the hyperparameters below, or replace the model_name
with a different continual learning algorithm from PyCIL (e.g., 'der', 'foster',
'memo', 'ewc', 'lwf', etc.).

To make deeper changes, agents can also modify models/icarl.py directly,
but must keep the BaseLearner interface (incremental_train, eval_task, after_task).
"""


_H007_PATCH_INSTALLED = False


def _install_classifier_weight_alignment_patch():
    """H007 minimal intervention (algorithm.py-only).

    Call the EXISTING `IncrementalNet.weight_align(increment)` exactly once
    per incremental stage, immediately after that stage's training finishes
    and before `trainer.py` calls `eval_task()` for the stage. Nothing else is
    changed: same iCaRL class, loss, optimizer, schedule, memory and exemplar
    selection. `weight_align` rescales the new-class rows of the classifier
    weight matrix so their mean L2 norm matches the old-class rows, which is
    the standard WA-style linear-head calibration.

    The wrapper must target `iCaRL.incremental_train` itself, not
    `BaseLearner.incremental_train`: iCaRL overrides that method, so wrapping
    the base-class attribute would never be reached (an earlier revision of
    this patch wrapped the base class and was provably inert -- stage 1 was
    bit-identical to H001 and no `alignweights,gamma=` line was printed).

    The base stage (task 0) is skipped because it has no old classes, i.e.
    aligning there would be undefined.
    """
    global _H007_PATCH_INSTALLED
    if _H007_PATCH_INSTALLED:
        return

    import torch
    from models.icarl import iCaRL

    original_incremental_train = iCaRL.incremental_train

    def _incremental_train_with_weight_align(self, data_manager):
        result = original_incremental_train(self, data_manager)
        if self._cur_task <= 0:
            return result
        network = self._network
        if isinstance(network, torch.nn.DataParallel):
            network = network.module
        # At this point _known_classes still holds the previous stage total
        # (BaseLearner.after_task updates it only after eval_task), so this is
        # exactly the increment of the stage that just trained.
        network.weight_align(self._total_classes - self._known_classes)
        return result

    iCaRL.incremental_train = _incremental_train_with_weight_align
    _H007_PATCH_INSTALLED = True


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
    # H007: post-training new-class classifier weight-norm alignment.
    _install_classifier_weight_alignment_patch()
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
