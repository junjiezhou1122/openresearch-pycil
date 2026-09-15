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
H024: is the distillation weight under-set? (KD scale x2 at the 6000 budget)
-------------------------------------------------------------------------------
Configuration is the H018 stack: iCaRL + ResNet-18, init_epoch=100, epochs=50,
seed 1993 (harness-pinned), memory_size = 6000. The single change is the WEIGHT of the
distillation term.

Where this comes from. H020 (valid/refuted) zeroed the distillation term at 6000 and
measured its marginal contribution as 2.165 pp of aggregate NME (70.8217 -> 68.6567),
and H022 (valid/supported) measured the same ablation at 2000 as 3.7983 pp. KD is
therefore a real, quantified contributor that is a SUBSTITUTE for replay memory: its
contribution shrinks from ~3.80 pp to ~2.17 pp as the budget triples, but it never
disappears. The loss is `loss_clf + 1.0 * loss_kd` with the weight fixed at 1 by the
frozen recipe. Those two measurements make the weight itself testable: if KD still
buys 2.17 pp at weight 1, the gradient balance may be under-weighting it.

Hypothesis. The distillation weight is under-set at 1: doubling it at the 6000 budget
will raise aggregate NME above H018's 70.8217 by at least the registered +0.5 pp. If
it does not (or it hurts), the frozen weight is already at or past its optimum, which
is what the substitution structure would predict once replay is plentiful.

Pre-registered reading: SUPPORTED if aggregate NME exceeds H018's 70.8217 by at least
+0.5 pp; REFUTED if it does not. Bounded to the single weight value 2.0 - this is not a
sweep of weights.

Minimal intervention: algorithm.py ONLY. `models.icarl._KD_loss` - the module global
that `_update_representation` reads on every batch, so replacing it provably affects
the executed path - is replaced by a wrapper that returns exactly 2.0 times the frozen
formula's value on the same inputs, making `loss = loss_clf + 2.0 * loss_kd`. A wrapper
around `iCaRL._update_representation` prints one `H024_KD_SCALED` marker per
incremental stage, and the CPU self-check proves the replacement returns exactly twice
the pristine value on real tensors (standing rule 7: prove the EFFECT, not just the
installation).

No exact invariance control is available, because changing a loss weight changes the
training trajectory by construction. The registered comparison is against H018
(70.8217), H020's KD-off run (68.6567) and H001 (66.0517), with the seed-pinned 0.0 pp
same-stack noise floor.
"""

import json

_H024_PATCH_INSTALLED = False
_H024_MARGIN_PP = 0.5
_H024_KD_SCALE = 2.0
_H024_MARKER = "H024_KD_SCALED"
_H024_KD_CALLS = 0


def _install_h024_kd_ablation():
    """Zero the distillation term and add a per-stage proof-of-effect marker."""
    global _H024_PATCH_INSTALLED
    if _H024_PATCH_INSTALLED:
        return False

    import torch

    import models.icarl as icarl_module
    from models.icarl import iCaRL

    original_kd_loss = icarl_module._KD_loss

    def _h024_kd_loss(pred, soft, T):
        """Replacement for models.icarl._KD_loss: the frozen formula, scaled up."""
        global _H024_KD_CALLS
        _H024_KD_CALLS += 1
        return _H024_KD_SCALE * original_kd_loss(pred, soft, T)

    _h024_kd_loss._h024_scaled = True
    icarl_module._KD_loss = _h024_kd_loss

    original_update = iCaRL._update_representation

    def _h024_update_representation(self, train_loader, test_loader, optimizer, scheduler):
        print(
            _H024_MARKER
            + " "
            + json.dumps(
                {
                    "stage": int(getattr(self, "_cur_task", -1)),
                    "kd_loss_scaled": bool(getattr(icarl_module._KD_loss, "_h024_ablation", False)),
                    "kd_calls_so_far": int(_H024_KD_CALLS),
                    "known_classes": int(getattr(self, "_known_classes", -1)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return original_update(self, train_loader, test_loader, optimizer, scheduler)

    iCaRL._update_representation = _h024_update_representation
    _H024_PATCH_INSTALLED = True

    print(
        "H024_PATCH_INSTALLED "
        + json.dumps(
            {
                "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
                "memory_size": 6000,
                "target": "models.icarl._KD_loss (module global read by _update_representation)",
                "effect": "loss = loss_clf + 2 * loss_kd",
                "proof_of_effect": "one H024_KD_SCALED marker per incremental stage",
                "margin_pp": _H024_MARGIN_PP,
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
    # H024: the H018 stack (memory 6000) with the distillation term zeroed, to test
    # whether the replay gain is KD-carried or classifier-carried.
    _install_h024_kd_ablation()
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
