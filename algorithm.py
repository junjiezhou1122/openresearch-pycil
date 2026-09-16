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
H027: Feature-Level Cosine Distillation to Mitigate Representation Drift
-------------------------------------------------------------------------------
H026 established that continual learning suffers a 5.55 pp representation drift
gap (76.37% Joint Oracle vs 70.8217% at memory 6000) that memory quantity and
readout geometry cannot close.

In standard iCaRL, distillation is applied solely to the output classification
logits via `_KD_loss`. The intermediate 512-dimensional feature representation
is unconstrained, allowing feature angles to rotate and drift across incremental
stages even when output logits on the replay exemplars are matched.

H027 adds an explicit feature-level cosine distillation loss:
    loss_feat = (1.0 - F.cosine_similarity(cur_features, old_features, dim=-1)).mean()
    loss = loss_clf + loss_kd + 1.0 * loss_feat

Motivation & Competing Mechanisms:
  (A) Feature drift hypothesis: Directly penalizing feature angular drift
      preserves intermediate geometric structure for old classes, reducing
      catastrophic representation forgetting and raising aggregate NME above
      H018's 70.8217.
  (B) Feature rigidity hypothesis: Penalizing feature drift over-constrains
      the network, impairing new-class plasticity without recovering old-class
      accuracy, leaving aggregate NME flat or worse.

Claim: Adding feature-level cosine distillation with weight 1.0 on the H018
stack (memory 6000) raises aggregate NME above H018's 70.8217 by >= +0.5 pp.
Pre-registered reading:
  - SUPPORTED: aggregate NME >= 71.3217 (+0.5 pp above H018).
  - REFUTED: valid run with aggregate NME < 71.3217.
  - INVALID: feature KD markers absent, RNG corrupted, or run fails.
"""

_H027_PATCH_INSTALLED = False


def _install_feature_cosine_distillation_patch():
    global _H027_PATCH_INSTALLED
    if _H027_PATCH_INSTALLED:
        return

    import logging
    from tqdm import tqdm
    import numpy as np
    import torch
    from torch.nn import functional as F
    import models.icarl as icarl_module
    from utils.toolkit import tensor2numpy

    def _h027_update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(icarl_module.epochs))
        feat_loss_sum = 0.0
        batch_count = 0

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                cur_out = self._network(inputs)
                old_out = self._old_network(inputs)

                logits = cur_out["logits"]
                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = icarl_module._KD_loss(
                    logits[:, : self._known_classes],
                    old_out["logits"],
                    icarl_module.T,
                )

                # Feature-level cosine distillation
                cur_feat = cur_out["features"]
                old_feat = old_out["features"]
                loss_feat = (1.0 - F.cosine_similarity(cur_feat, old_feat, dim=-1)).mean()

                loss = loss_clf + loss_kd + 1.0 * loss_feat

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                feat_loss_sum += loss_feat.item()
                batch_count += 1

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    icarl_module.epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    icarl_module.epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        avg_feat_loss = feat_loss_sum / max(1, batch_count)
        marker = (
            f"[H027_FEAT_KD] Task {self._cur_task} finished: "
            f"batches={batch_count}, avg_feat_loss={avg_feat_loss:.4f}, "
            f"known_classes={self._known_classes}, total_classes={self._total_classes}"
        )
        print(marker, flush=True)
        logging.info(marker)
        logging.info(info)

    icarl_module.iCaRL._update_representation = _h027_update_representation
    _H027_PATCH_INSTALLED = True
    print("H027_PATCH_INSTALLED: feature cosine distillation patch installed on iCaRL._update_representation", flush=True)


def get_pycil_config():
    """
    Return PyCIL experiment configuration.

    H027 stack: Memory 6000 (H018 incumbent) + Feature Cosine Distillation (weight=1.0).
    """
    _install_feature_cosine_distillation_patch()
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
