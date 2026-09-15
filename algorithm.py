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


# ---------------------------------------------------------------------------
# H018: the third budget point - is the memory->NME response monotone or saturating?
#
# Wave 3 established the exemplar budget as the only lever that has ever moved the
# accepted metric, and wave 4 closed the selection axis:
#   * H013 (valid/supported): memory_size 2000 -> 4000 took aggregate NME from
#     0.6605167 to 0.6881833 (+2.7666 pp), entirely old-class retention.
#   * H014 (valid/refuted): that gain is replay/representation-side, not prototype-side.
#   * H015 (valid/supported, +1.0000 pp) and H016 (valid/supported, +1.165 pp,
#     independent draw): herding beats RANDOM selection at the same budget, so
#     ~1.1 pp of the gain is the stored set's information content.
#   * H017 (valid/refuted): greedy k-center is 3.00 pp WORSE than herding and below
#     both random draws, so coverage is the wrong axis and herding is at the ceiling.
#
# With the selection axis closed, the only open question on this lever is its SHAPE.
# H018 is therefore the third point of the budget->NME response curve and is again a
# ONE-NUMBER intervention: raise memory_size 4000 -> 6000 (120 -> 60 exemplars per
# class across the six stages). Everything else is byte-identical to H001/H013.
#
# Pre-registered reading: if the response is still rising materially at 4000, the
# 6000 point must add at least +1.0 pp of aggregate NME over H013's 68.8183 (a
# quarter of the 2000->4000 step, and far above the 0.0 pp same-stack deterministic
# noise floor); if the response has saturated, the increment is smaller.
#
# Because this changes the training set (more replay data), no exact invariance
# control is available; the registered comparison is against H013's 4000 point and
# H001's 2000 point, with the measured 0.0 pp noise floor for same-stack
# reported-precision comparisons.
# ---------------------------------------------------------------------------


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
        "memory_size": 6000,   # H018: third budget point (baseline 2000, H013 = 4000)
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
