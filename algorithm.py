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
H022: are distillation and replay SUBSTITUTES or COMPLEMENTS?
-------------------------------------------------------------------------------
Configuration is the H001 incumbent stack: iCaRL + ResNet-18, init_epoch=100,
epochs=50, seed 1993 (harness-pinned), memory_size = 2000. The intervention is the
same KD ablation as H020.

Where this comes from. H020 (valid/refuted) zeroed the distillation term at the
H018 6000 budget and found that it costs only 2.165 pp of aggregate NME
(70.8217 -> 68.6567): the budget's replay gain is majority classifier-carried, with
distillation a minority contribution. That single point cannot say whether KD's
contribution is a CONSTANT, or whether it depends on how much replay memory the
learner already has. The classic continual-learning hypothesis is that distillation
matters most when replay is scarce, i.e. that the two are SUBSTITUTES. This is the
other cell of the 2x2:

                       KD on            KD off
  memory 2000     66.0517 (H001)     this experiment (H022)
  memory 6000     70.8217 (H018)     68.6567 (H020, cost 2.165 pp)

Hypothesis (substitutes). Zeroing KD costs MORE at the 2000 budget than the 2.165 pp
it costs at 6000, by at least the registered 1.0 pp margin, i.e. KD-off at 2000 lands
at or below 64.89 aggregate NME. If the cost at 2000 is smaller, distillation's
contribution does not shrink as memory grows and the two channels are not
substitutes under this protocol.

Minimal intervention: algorithm.py ONLY, base 839aece. The H001 configuration is
unchanged (memory_size 2000, every other value H001) and the single behavioural change
is that `models.icarl._KD_loss` - which `_update_representation` reads as a module
global on every batch, so replacing it provably affects the executed path - returns a
zero scalar, making `loss = loss_clf + loss_kd` equal `loss = loss_clf`. A wrapper
around `iCaRL._update_representation` prints one `H022_KD_ABLATED` marker per
incremental stage.

Proof of effect (standing rule 7): the run's own stdout must show an ablation marker
with rising ablated-call counters at every incremental stage, and the CPU self-check
additionally proves the replacement returns exactly zero on real tensors while the
pristine `_KD_loss` returns a strictly positive value on the same inputs.

No exact invariance control is available, because removing a loss term changes the
training trajectory by construction; the registered comparison is against H001
(66.0517), H018 (70.8217) and H020 (68.6567), with the seed-pinned 0.0 pp same-stack
noise floor.
"""

import json

_H022_PATCH_INSTALLED = False
_H022_MARGIN_PP = 1.0
_H022_MARKER = "H022_KD_ABLATED"
_H022_KD_CALLS = 0


def _install_h022_kd_ablation():
    """Zero the distillation term and add a per-stage proof-of-effect marker."""
    global _H022_PATCH_INSTALLED
    if _H022_PATCH_INSTALLED:
        return False

    import torch

    import models.icarl as icarl_module
    from models.icarl import iCaRL

    def _h022_kd_loss(pred, soft, T):
        """Replacement for models.icarl._KD_loss: returns a zero scalar."""
        global _H022_KD_CALLS
        _H022_KD_CALLS += 1
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    _h022_kd_loss._h022_ablation = True
    icarl_module._KD_loss = _h022_kd_loss

    original_update = iCaRL._update_representation

    def _h022_update_representation(self, train_loader, test_loader, optimizer, scheduler):
        print(
            _H022_MARKER
            + " "
            + json.dumps(
                {
                    "stage": int(getattr(self, "_cur_task", -1)),
                    "kd_loss_replaced": bool(getattr(icarl_module._KD_loss, "_h022_ablation", False)),
                    "kd_calls_so_far": int(_H022_KD_CALLS),
                    "known_classes": int(getattr(self, "_known_classes", -1)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return original_update(self, train_loader, test_loader, optimizer, scheduler)

    iCaRL._update_representation = _h022_update_representation
    _H022_PATCH_INSTALLED = True

    print(
        "H022_PATCH_INSTALLED "
        + json.dumps(
            {
                "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
                "memory_size": 2000,
                "target": "models.icarl._KD_loss (module global read by _update_representation)",
                "effect": "loss = loss_clf + 0",
                "proof_of_effect": "one H022_KD_ABLATED marker per incremental stage",
                "margin_pp": _H022_MARGIN_PP,
                "training_otherwise_untouched": True,
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
    # H022: the H018 stack (memory 6000) with the distillation term zeroed, to test
    # whether the replay gain is KD-carried or classifier-carried.
    _install_h022_kd_ablation()
    return {
        "prefix": "benchmark",
        "dataset": "cifar100",
        "memory_size": 2000,   # H022: the H001 incumbent budget
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
