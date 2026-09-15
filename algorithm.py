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
H020: which loss inside the replay stream carries the budget gain - KD or clf?
-------------------------------------------------------------------------------
Configuration is the H018 stack: iCaRL + ResNet-18, init_epoch=100, epochs=50,
seed 1993 (harness-pinned), memory_size = 6000. Nothing about training, memory,
exemplar selection or the evaluator differs from H018.

Where this comes from. Wave 3 and 4 established that the exemplar budget is the only
lever that has moved the accepted metric (+2.7666 pp for 2000 -> 4000 in H013, a
further +2.0033 pp for 4000 -> 6000 in H018; cumulative +4.77 pp), and H014 showed
the gain is replay/representation-side rather than prototype-side (rolling the NME
prototype table back to H001's per-class budget costs only 0.0483 pp). But iCaRL's
`_update_representation` feeds the SAME enlarged replay batch into TWO losses:
the classifier term (`loss_clf`, cross-entropy on the current-stage classes) and the
distillation term (`loss_kd`, KD over the old-class logits against the frozen
teacher). H014 localised the gain to "the replay stream" but cannot say which loss
converts extra exemplars into old-class retention, and H018's gain is again almost
entirely old-class.

Hypothesis. The replay gain is carried by the DISTILLATION term: with KD removed at
the 6000 budget the old-class retention that the extra exemplars buy should largely
vanish, so aggregate NME should fall well below H018's 70.8217.

Pre-registered reading: SUPPORTED if zeroing KD reduces aggregate NME by at least
3.0 pp from H018's 70.8217 (i.e. it removes most of the +4.77 pp budget gain);
REFUTED if the reduction is smaller than 3.0 pp, in which case the extra replay data
is working mainly through the classifier term. The 3.0 pp margin is ~63% of the
cumulative budget gain and far above the seed-pinned 0.0 pp same-stack noise floor.

Minimal intervention: algorithm.py ONLY. The H018 configuration is unchanged
(memory_size 6000, every other value H001) and the single behavioural change is that
`models.icarl._KD_loss` - which `_update_representation` looks up as a module global
on every batch, so replacing it provably affects the executed code path - is replaced
by a function returning a zero scalar, making `loss = loss_clf + loss_kd` equal to
`loss_clf`. A wrapper around `iCaRL._update_representation` prints one
`H020_KD_ABLATED stage=<n>` marker per incremental stage so the ablation is provably
active (this is the check H019 lacked: the intervention's EFFECT must be observed in
the run, not merely its installation).

No exact invariance control is available, because removing a loss term changes the
training trajectory by construction; the registered comparison is against H018's
6000-budget run (aggregate 70.8217) and H001 (0.6605167) with the measured 0.0 pp
same-stack noise floor.

Operational note carried from H019: the frozen evaluator supplies/overrides the
run configuration, so a candidate that only changes a config VALUE the evaluator
also sets (the seed) never fires. Proof of effect must come from the run's own
stdout, which is why this hypothesis logs a per-stage ablation marker.
"""

import json

_H020_PATCH_INSTALLED = False
_H020_MARGIN_PP = 3.0
_H020_MARKER = "H020_KD_ABLATED"
_H020_KD_CALLS = 0


def _install_h020_kd_ablation():
    """Zero the distillation term and add a per-stage proof-of-effect marker."""
    global _H020_PATCH_INSTALLED
    if _H020_PATCH_INSTALLED:
        return False

    import torch

    import models.icarl as icarl_module
    from models.icarl import iCaRL

    def _h020_kd_loss(pred, soft, T):
        """Replacement for models.icarl._KD_loss: returns a zero scalar."""
        global _H020_KD_CALLS
        _H020_KD_CALLS += 1
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    _h020_kd_loss._h020_ablation = True
    icarl_module._KD_loss = _h020_kd_loss

    original_update = iCaRL._update_representation

    def _h020_update_representation(self, train_loader, test_loader, optimizer, scheduler):
        print(
            _H020_MARKER
            + " "
            + json.dumps(
                {
                    "stage": int(getattr(self, "_cur_task", -1)),
                    "kd_loss_replaced": bool(getattr(icarl_module._KD_loss, "_h020_ablation", False)),
                    "kd_calls_so_far": int(_H020_KD_CALLS),
                    "known_classes": int(getattr(self, "_known_classes", -1)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return original_update(self, train_loader, test_loader, optimizer, scheduler)

    iCaRL._update_representation = _h020_update_representation
    _H020_PATCH_INSTALLED = True

    print(
        "H020_PATCH_INSTALLED "
        + json.dumps(
            {
                "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
                "memory_size": 6000,
                "target": "models.icarl._KD_loss (module global read by _update_representation)",
                "effect": "loss = loss_clf + 0",
                "proof_of_effect": "one H020_KD_ABLATED marker per incremental stage",
                "margin_pp": _H020_MARGIN_PP,
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
    # H020: the H018 stack (memory 6000) with the distillation term zeroed, to test
    # whether the replay gain is KD-carried or classifier-carried.
    _install_h020_kd_ablation()
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
