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
H028: Plasticity-Stability Regularization via Lower-Layer Freezing
-------------------------------------------------------------------------------
H026 established that continual learning incurs an exact 5.55 pp representation
drift tax against the Offline Joint Oracle (70.8217% vs 76.37%).

In deep residual networks on small images (CIFAR-100, 32x32), the earliest layers
(Conv1: 3x3 conv, and Layer1: 2 BasicBlocks) learn generic, low-level visual
primitives (edge detectors, color gradients, texture filters). In Stage 0,
they are trained on 50 diverse classes (25,000 images), providing a rich sensory
foundation.

During incremental stages 1-5, gradient updates backpropagate through the entire
network. Even slight perturbations to low-level weights cause cascading drift in
penultimate feature representations for old classes.

H028 implements Plasticity-Stability Regularization by freezing Conv1 and Layer1
after Stage 0:
  - In Stage 0: Conv1 and Layer1 train normally on the 50 base classes.
  - After Stage 0: Conv1 and Layer1 parameters are frozen (requires_grad=False)
    and their BatchNorm layers are locked in eval mode.
  - In Stages 1-5: Only Layer2, Layer3, Layer4, and the classifier adapt.

Claim: Freezing Conv1 and Layer1 after Stage 0 on the H018 stack (memory 6000)
preserves the low-level representation foundation, raising aggregate NME above
H018's 70.8217 by >= +0.5 pp (aggregate NME >= 71.3217).
"""

_H028_PATCH_INSTALLED = False


def _install_lower_layer_freezing_patch():
    global _H028_PATCH_INSTALLED
    if _H028_PATCH_INSTALLED:
        return

    import logging
    import torch
    import torch.nn as nn
    import models.icarl as icarl_module

    original_after_task = icarl_module.iCaRL.after_task

    def _h028_after_task(self):
        original_after_task(self)
        if self._cur_task == 0:
            network = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
            # Freeze conv1
            for p in network.convnet.conv1.parameters():
                p.requires_grad = False
            network.convnet.conv1.eval()
            network.convnet.conv1.train = lambda mode=True: network.convnet.conv1

            # Freeze layer1
            for p in network.convnet.layer1.parameters():
                p.requires_grad = False
            network.convnet.layer1.eval()
            network.convnet.layer1.train = lambda mode=True: network.convnet.layer1

            # Verify and log parameter counts
            total_params = sum(p.numel() for p in network.parameters())
            trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params

            marker = (
                f"[H028_LAYER_FREEZE] Task 0 completed: Conv1 and Layer1 frozen! "
                f"total={total_params}, trainable={trainable_params}, frozen={frozen_params}"
            )
            print(marker, flush=True)
            logging.info(marker)

    icarl_module.iCaRL.after_task = _h028_after_task
    _H028_PATCH_INSTALLED = True
    print("H028_PATCH_INSTALLED: lower-layer freezing patch installed on iCaRL.after_task", flush=True)


def get_pycil_config():
    """
    Return PyCIL experiment configuration.

    H028 stack: Memory 6000 (H018 incumbent) + Conv1/Layer1 Freezing after Task 0.
    """
    _install_lower_layer_freezing_patch()
    return {
        "prefix": "benchmark",
        "dataset": "cifar100",
        "memory_size": 6000,
        "memory_per_class": 20,
        "fixed_memory": False,
        "shuffle": True,
        "init_cls": 50,
        "increment": 10,
        "model_name": "icarl",
        "convnet_type": "resnet18",
        "device": ["0"],
        "seed": [1993],

        # Standard settings
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
