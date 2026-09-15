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
# H013: memory-budget response experiment (wave 3, search space "memory-budget").
#
# Wave 2 identified the binding constraint POSITIVELY rather than by elimination:
#   * H010 (valid/refuted) and H012 (valid/refuted) showed that no closed-form
#     readout fitted on the learner's real information set beats the registered
#     nearest-mean rule (-0.66 pp and -0.67 pp aggregate), so the registered rule
#     already sits at the learner's information-set frontier.
#   * H011 (valid/supported) showed a full-train oracle DOES beat it at every
#     incremental stage (+0.94/+2.66/+3.13/+3.67/+3.77 pp; +2.36 pp aggregate),
#     with the entire gain on OLD classes (+2.3..+5.2 pp) and new-class accuracy
#     regressing (-6.0..-11.3 pp).
#
# So the +2.36 pp of headroom exists in the frozen features but requires more
# OLD-class data than the 2000-exemplar budget holds. H013 tests that directly
# and is deliberately a ONE-NUMBER intervention: raise memory_size 2000 -> 4000
# (20 -> 40 exemplars/class at the final stage; 80/class at the base stage).
# Everything else is byte-identical to the frozen H001 configuration.
#
# This is a resource lever identified BY evidence, not a blind parameter sweep.
# Because it changes the training set (more exemplars enter KD and the classifier
# loss), no exact invariance control is available; the registered comparison is
# against H001 and the H011 oracle bound, with the measured 0.0 pp H004 noise
# floor for the fixed-stack comparisons.
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
        "memory_size": 4000,   # H013: doubled exemplar budget (was 2000)
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
