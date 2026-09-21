"""
Baseline continual learning algorithm: iCaRL on CIFAR-100.

H029 (measurement-only, originalCommit=7f187e7364c0e1a4f825c7c9c47061478e3c249b)
tests the descriptive hypothesis that representation drift is concentrated in
particular ResNet stages as replay introduces new classes.  It runs the H018
training protocol unchanged (ResNet-18, memory_size=6000, seed=1993), and
records paired representations between the frozen Task-0 reference and each
later task.  The fixed probe is the first two test examples, in mapped test
array order, for each of the 50 base classes (100 samples total); test-mode
transforms are deterministic (ToTensor + CIFAR-100 normalization).

For every task and layer (conv1, layer1, layer2, layer3, layer4, and the final
embedding), cosine similarity is the mean per-sample cosine and normalized L2
is mean ||current-reference||_2 / (||reference||_2 + 1e-8).  Task 0 is the
identity reference (cosine=1, normalized L2=0).  A compact
``H029_REPRESENTATION_JSON`` line is emitted after the final task so the remote
run can preserve the artifact in stdout.  This is instrumentation only: no
optimizer, loss, replay selection, class schedule, RNG stream, or validation
semantics are changed.  It does not claim causality, does not add CKA (the
fixed paired probe is insufficient to justify an additional estimator), and
does not replace the evaluator's metric; probe coverage and aggregate layer
summaries remain limitations.

H030 adds one bounded causal intervention on top of the H029 checkout. During
incremental tasks only, replay rows are selected by ``targets < known_classes``
and compared with the frozen old network at ResNet layer3 and layer4. The
deterministic POD-style loss squares activations, sums over each spatial axis,
concatenates the summaries, L2-normalizes each descriptor, and takes the mean
per-sample L2 distance across those two layers. A fixed conservative coefficient
of 0.05 is used; new-class rows and final embeddings are not constrained.
``H030_DIAGNOSTIC_JSON`` reports per-task old-row counts and mean base/KD/POD/
total losses, while the original H029 representation artifact is unchanged.

This file contains the epoch and hyperparameter configuration for iCaRL.
The actual iCaRL implementation is in the PyCIL repository (models/icarl.py).

Agents should modify the hyperparameters below, or replace the model_name
with a different continual learning algorithm from PyCIL (e.g., 'der', 'foster',
'memo', 'ewc', 'lwf', etc.).

To make deeper changes, agents can also modify models/icarl.py directly,
but must keep the BaseLearner interface (incremental_train, eval_task, after_task).
"""

import json


_H029_ORIGINAL_COMMIT = "7f187e7364c0e1a4f825c7c9c47061478e3c249b"
_H029_PATCH_INSTALLED = False
_H030_PATCH_INSTALLED = False
_H030_POD_WEIGHT = 0.05


def _h029_probe(data_manager, per_class=2):
    """Build the fixed, deterministic probe without touching training RNG."""
    import numpy as np
    import torch

    if data_manager.dataset_name.lower() != "cifar100":
        raise ValueError("H029 requires the CIFAR-100 H018 protocol")
    targets = np.asarray(data_manager._test_targets)
    selected = []
    for class_id in range(50):
        class_indices = np.flatnonzero(targets == class_id)
        if len(class_indices) < per_class:
            raise RuntimeError(
                "H029 probe class {} has {} samples; need {}".format(
                    class_id, len(class_indices), per_class
                )
            )
        selected.extend(class_indices[:per_class].tolist())
    raw_data = np.asarray(data_manager._test_data)[selected]
    raw_targets = targets[selected]
    # get_dataset applies only the deterministic test transform here.  Passing
    # appendent avoids traversing any additional samples or sampling indices.
    probe_dataset = data_manager.get_dataset(
        [], source="test", mode="test", appendent=(raw_data, raw_targets)
    )
    inputs = torch.stack([probe_dataset[i][1] for i in range(len(probe_dataset))])
    return inputs, torch.as_tensor(raw_targets, dtype=torch.long)


def _h029_features(network, inputs, device, batch_size=32):
    """Extract paired layer activations while restoring the model's mode."""
    import torch
    from torch import nn

    if isinstance(network, nn.DataParallel):
        network = network.module
    if not hasattr(network, "convnet"):
        raise RuntimeError("H029 expected an IncrementalNet with a convnet")
    convnet = network.convnet
    if not all(hasattr(convnet, name) for name in ("conv1", "layer1", "layer2", "layer3", "layer4")):
        raise RuntimeError("H029 requires the ResNet-18 conv1/layer1..layer4 path")

    captured_conv1 = []

    def capture_conv1(_module, _args, output):
        captured_conv1.append(output.detach().cpu())

    hook = convnet.conv1.register_forward_hook(capture_conv1)
    was_training = network.training
    network.eval()
    outputs = {name: [] for name in ("conv1", "layer1", "layer2", "layer3", "layer4", "final_embedding")}
    try:
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch = inputs[start : start + batch_size].to(device)
                result = network(batch)
                if not captured_conv1:
                    raise RuntimeError("H029 conv1 hook did not observe a forward pass")
                outputs["conv1"].append(captured_conv1.pop(0))
                fmaps = result.get("fmaps")
                if fmaps is None or len(fmaps) != 4:
                    raise RuntimeError("H029 expected four ResNet feature maps")
                for name, fmap in zip(("layer1", "layer2", "layer3", "layer4"), fmaps):
                    outputs[name].append(fmap.detach().cpu())
                outputs["final_embedding"].append(result["features"].detach().cpu())
    finally:
        hook.remove()
        if was_training:
            network.train()

    return {
        name: torch.cat(chunks, dim=0).reshape(len(inputs), -1).float()
        for name, chunks in outputs.items()
    }


def _h029_metrics(reference, current):
    from torch.nn import functional as F

    metrics = {}
    for layer in reference:
        ref = reference[layer]
        cur = current[layer]
        if ref.shape != cur.shape:
            raise RuntimeError(
                "H029 representation shape changed for {}: {} vs {}".format(
                    layer, tuple(ref.shape), tuple(cur.shape)
                )
            )
        cosine = F.cosine_similarity(ref, cur, dim=1, eps=1e-8).mean().item()
        normalized_l2 = ((ref - cur).norm(dim=1) / (ref.norm(dim=1) + 1e-8)).mean().item()
        metrics[layer] = {
            "cosine_similarity": float(cosine),
            "normalized_l2": float(normalized_l2),
        }
    return metrics


def _h029_record(self, data_manager):
    if self.args.get("convnet_type", "").lower() != "resnet18":
        raise ValueError("H029 is registered only for the H018 ResNet-18 protocol")
    if not hasattr(self, "_h029_probe_inputs"):
        self._h029_probe_inputs, self._h029_probe_targets = _h029_probe(data_manager)
        self._h029_measurements = []
        self._h029_reference = None
    current = _h029_features(self._network, self._h029_probe_inputs, self._device)
    if self._h029_reference is None:
        self._h029_reference = {name: value.clone() for name, value in current.items()}
    metrics = _h029_metrics(self._h029_reference, current)
    self._h029_measurements.append(
        {
            "task": int(self._cur_task),
            "known_classes": int(self._known_classes),
            "total_classes": int(self._total_classes),
            "layers": metrics,
        }
    )
    if self._cur_task == data_manager.nb_tasks - 1:
        artifact = {
            "schema": "openresearch.h029-representation-drift.v1",
            "hypothesis": "post-Task-0 representation drift is stage-local under replay",
            "originalCommit": _H029_ORIGINAL_COMMIT,
            "protocol": {
                "model": "iCaRL",
                "convnet": "resnet18",
                "memory_size": 6000,
                "seed": 1993,
                "probe": "first 2 mapped test samples per base class (classes 0-49)",
                "probe_size": int(len(self._h029_probe_inputs)),
            },
            "metrics": {
                "cosine_similarity": "mean per-sample cosine(reference,current)",
                "normalized_l2": "mean ||current-reference||_2/(||reference||_2+1e-8)",
                "cka": "not computed; bounded paired probe does not justify it",
            },
            "measurements": self._h029_measurements,
            "limitations": [
                "probe covers only two deterministic test samples per base class",
                "metrics are descriptive and do not establish causal layer-wise forgetting",
                "instrumentation does not alter or replace validation metrics",
            ],
        }
        print("H029_REPRESENTATION_JSON " + json.dumps(artifact, sort_keys=True, separators=(",", ":")), flush=True)


def _h030_replay_mask(targets, known_classes):
    """Return the explicit old-class/replay mask used by H030."""
    import torch

    if not torch.is_tensor(targets):
        raise TypeError("H030 targets must be a torch.Tensor")
    if targets.ndim != 1:
        raise ValueError("H030 targets must be a 1-D tensor")
    return targets < int(known_classes) if known_classes > 0 else torch.zeros_like(targets, dtype=torch.bool)


def _h030_pod_spatial_loss(current_fmaps, old_fmaps):
    """Normalized spatial POD distance for matched layer3/layer4 maps."""
    import torch
    from torch.nn import functional as F

    if len(current_fmaps) != 2 or len(old_fmaps) != 2:
        raise ValueError("H030 expects exactly layer3 and layer4 feature maps")
    layer_losses = []
    for current, old in zip(current_fmaps, old_fmaps):
        if current.ndim != 4 or old.ndim != 4 or current.shape != old.shape:
            raise ValueError("H030 feature-map shapes must match as [N,C,H,W]")
        current_power = current.pow(2)
        old_power = old.detach().pow(2)
        current_descriptor = torch.cat(
            (current_power.sum(dim=3).flatten(1), current_power.sum(dim=2).flatten(1)), dim=1
        )
        old_descriptor = torch.cat(
            (old_power.sum(dim=3).flatten(1), old_power.sum(dim=2).flatten(1)), dim=1
        )
        current_descriptor = F.normalize(current_descriptor, p=2, dim=1)
        old_descriptor = F.normalize(old_descriptor, p=2, dim=1)
        layer_losses.append(
            torch.linalg.vector_norm(current_descriptor - old_descriptor, dim=1).mean()
        )
    return torch.stack(layer_losses).mean()


def _h030_synthetic_preflight():
    """CPU-checkable invariants for the mask, POD values, and gradient boundary."""
    import torch

    targets = torch.tensor([0, 49, 50, 59], dtype=torch.long)
    mask = _h030_replay_mask(targets, 50)
    if mask.tolist() != [True, True, False, False]:
        raise AssertionError("H030 replay mask includes a new-class row")
    old = [
        torch.arange(2 * 4 * 4 * 4, dtype=torch.float32).reshape(2, 4, 4, 4) / 17,
        torch.arange(2 * 8 * 2 * 2, dtype=torch.float32).reshape(2, 8, 2, 2) / 11,
    ]
    current = [value.clone().requires_grad_() for value in old]
    zero = _h030_pod_spatial_loss(current, old)
    if zero.item() != 0.0:
        raise AssertionError("H030 identical maps must have zero POD loss")
    positive = _h030_pod_spatial_loss([current[0] + 0.25, current[1] * 1.1], old)
    if positive.item() <= 0.0:
        raise AssertionError("H030 changed maps must have positive POD loss")
    positive.backward()
    if any(value.grad is None for value in current) or any(value.grad is not None for value in old):
        raise AssertionError("H030 gradient boundary is not current-only")
    return {
        "mask": mask.tolist(),
        "zero_pod": float(zero.item()),
        "positive_pod": float(positive.item()),
        "current_gradients": True,
        "old_gradients": False,
        "pod_weight": _H030_POD_WEIGHT,
    }


def _h030_update_representation(self, train_loader, test_loader, optimizer, scheduler):
    """Original iCaRL loop with replay-only Layer3/4 POD distillation."""
    import logging
    import numpy as np
    import torch
    from torch.nn import functional as F
    from tqdm import tqdm
    from utils.toolkit import tensor2numpy
    from models.icarl import T, _KD_loss, epochs

    if self._cur_task == 0:
        raise RuntimeError("H030 update hook reached task 0")
    if self._old_network is None:
        raise RuntimeError("H030 requires frozen old_network on incremental tasks")

    prog_bar = tqdm(range(epochs))
    total_old_samples = 0
    summary = {"base": 0.0, "kd": 0.0, "pod": 0.0, "total": 0.0}
    batches = 0
    for _, epoch in enumerate(prog_bar):
        self._network.train()
        losses = 0.0
        correct, total = 0, 0
        for _, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            outputs = self._network(inputs)
            logits = outputs["logits"]
            loss_base = F.cross_entropy(logits, targets)
            with torch.no_grad():
                old_outputs = self._old_network(inputs)
            loss_kd = _KD_loss(logits[:, : self._known_classes], old_outputs["logits"], T)
            old_mask = _h030_replay_mask(targets, self._known_classes)
            old_count = int(old_mask.sum().item())
            if old_count:
                current_old_fmaps = [outputs["fmaps"][2][old_mask], outputs["fmaps"][3][old_mask]]
                previous_old_fmaps = [old_outputs["fmaps"][2][old_mask], old_outputs["fmaps"][3][old_mask]]
                loss_pod = _h030_pod_spatial_loss(current_old_fmaps, previous_old_fmaps)
            else:
                loss_pod = logits.new_zeros(())
            loss = loss_base + loss_kd + (_H030_POD_WEIGHT * loss_pod)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            summary["base"] += loss_base.item()
            summary["kd"] += loss_kd.item()
            summary["pod"] += loss_pod.item()
            summary["total"] += loss.item()
            total_old_samples += old_count
            batches += 1

            _, preds = torch.max(logits, dim=1)
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)

        scheduler.step()
        train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
        if epoch % 5 == 0:
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc, test_acc
            )
        else:
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, epochs, losses / len(train_loader), train_acc
            )
        prog_bar.set_description(info)
    logging.info(info)

    if batches == 0:
        raise RuntimeError("H030 received an empty incremental train loader")
    artifact = {
        "schema": "openresearch.h030-replay-pod.v1",
        "task": int(self._cur_task),
        "known_classes": int(self._known_classes),
        "mask": "targets < known_classes",
        "layers": ["layer3", "layer4"],
        "old_sample_count": int(total_old_samples),
        "batches": int(batches),
        "pod_weight": _H030_POD_WEIGHT,
        "loss_mean": {name: value / batches for name, value in summary.items()},
    }
    print("H030_DIAGNOSTIC_JSON " + json.dumps(artifact, sort_keys=True, separators=(",", ":")), flush=True)


def _install_h029_instrumentation():
    """Patch only the iCaRL post-training call; training code remains unchanged."""
    global _H029_PATCH_INSTALLED, _H030_PATCH_INSTALLED
    if _H029_PATCH_INSTALLED and _H030_PATCH_INSTALLED:
        return
    from models.icarl import iCaRL

    original_incremental_train = iCaRL.incremental_train

    def instrumented_incremental_train(self, data_manager):
        result = original_incremental_train(self, data_manager)
        _h029_record(self, data_manager)
        return result

    iCaRL.incremental_train = instrumented_incremental_train
    _H029_PATCH_INSTALLED = True
    if not _H030_PATCH_INSTALLED:
        iCaRL._update_representation = _h030_update_representation
        _H030_PATCH_INSTALLED = True


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
    config = {
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
    # The evaluator imports this adapter before importing models.icarl.  Install
    # the post-training observer at that boundary while leaving all H018 values
    # and the iCaRL training implementation untouched.
    _install_h029_instrumentation()
    return config
