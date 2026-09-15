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
# H019: cross-SEED noise floor for the frozen H001 stack (wave 5)
#
# Why this is the first thing wave 5 must do. Every utility claim in this loop is
# a difference against the frozen H001 incumbent (aggregate NME 0.6605167):
#   * H013: memory 2000 -> 4000 gave +2.7666 pp
#   * H018: memory 4000 -> 6000 gave a further +2.0033 pp (cumulative +4.77 pp)
#   * H015/H016: herding beats random selection by 1.000 pp / 1.165 pp
#   * H017: k-center is 3.0033 pp WORSE than herding
# H004 established a 0.0 pp noise floor, but ONLY for IDENTICAL trees (an empty
# commit re-run): it bounds measurement/run-to-run variation, NOT variation across
# training seeds. Every one of the claims above rests on single-training-seed
# points, so the loop has been reading effects of 1-3 pp against a floor that was
# never measured for the perturbation a reader would actually worry about.
#
# H019 therefore measures the missing floor: re-run the frozen H001 configuration
# with ONE config value changed - seed 1993 -> 7919 - and compare every registered
# observable against H001. This is the cross-seed analogue of H004.
#
# Pre-registered reading: if the stack is seed-stable at reported precision (the
# aggregate NME moves by less than 0.5 pp), then the loop's 1-4.8 pp effects are
# comfortably above cross-seed variation and the utility claims stand. If the
# aggregate moves by 0.5 pp or more, then sub-1 pp claims (in particular the
# H015/H016 selection-content term) must be re-read against the new floor.
#
# Everything else is byte-identical to the frozen H001 configuration.
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
        "memory_size": 2000,
        "memory_per_class": 20,
        "fixed_memory": False,
        "shuffle": True,
        "init_cls": 50,
        "increment": 10,
        "model_name": "icarl",
        "convnet_type": "resnet18",
        "device": ["0"],
        "seed": [7919],   # H019: cross-seed calibration (H001 uses 1993)

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
