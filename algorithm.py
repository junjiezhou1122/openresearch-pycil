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
H026 representation-learning Oracle: Offline Joint Multitask Training Bound
-------------------------------------------------------------------------------
Wave 6 concluded by empirically and theoretically exhausting the replay memory
and readout geometry search spaces (H018 reached 70.8217% NME, and H021 proved
the full-train linear probe only reaches 71.36% with +0.54 pp headroom remaining).

To open the `representation-learning` search space and guide feature-level
distillation and regularisation interventions, H026 measures the true physical
upper bound of the feature extractor: Offline Joint Multitask Training.

By setting init_cls=100 and increment=0, all 100 CIFAR-100 classes are trained
jointly in a single stage without continual learning or catastrophic forgetting,
using the exact same ResNet-18 architecture, seed (1993), 100 epochs, and
optimiser schedule as the continual learning base stage.

The resulting accuracy defines the gold-standard Oracle bound (~76-78%), and
the delta (Joint Oracle minus 70.8217%) quantifies the exact "representation
drift tax" imposed by continual learning.
"""


def get_pycil_config():
    """
    Return PyCIL experiment configuration.

    H026: Offline Joint Multitask Training Oracle (100 classes in one stage).
    All architecture, seed, and base-stage training hyperparameters are identical to H001.
    """
    return {
        "prefix": "benchmark",
        "dataset": "cifar100",
        "memory_size": 2000,
        "memory_per_class": 20,
        "fixed_memory": False,
        "shuffle": True,
        "init_cls": 100,
        "increment": 0,
        "model_name": "icarl",
        "convnet_type": "resnet18",
        "device": ["0"],
        "seed": [1993],

        # Identical to H001 base-stage training
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
