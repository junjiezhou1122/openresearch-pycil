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

H032 adds one bounded intervention to that exact H018/H029 stack.  On
incremental tasks, only replay rows whose labels satisfy
``targets < known_classes`` receive a Layer4 spatial-POD target from the
detached old network.  The deterministic loss squares activations, pools them
along height and width, concatenates and L2-normalizes the descriptors, and
uses their mean per-sample L2 distance with coefficient 0.05.  New-class rows
and the final embedding are never constrained by H032.  A compact
``H032_DIAGNOSTIC_JSON`` records the intervention, per-task engagement, the
synthetic CPU contract, and the pre-registered interpretation thresholds.

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

_H032_PARENT_COMMIT = "12a420d8eeb34575599481cebd5831974ff8214c"
_H032_SPATIAL_COEFFICIENT = 0.05
_H032_PATCH_INSTALLED = False
_H032_CONTRACT_CHECKED = False


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
        _h032_emit_diagnostic(self, data_manager)


def _install_h029_instrumentation():
    """Patch only the iCaRL post-training call; training code remains unchanged."""
    global _H029_PATCH_INSTALLED
    if _H029_PATCH_INSTALLED:
        return
    from models.icarl import iCaRL

    original_incremental_train = iCaRL.incremental_train

    def instrumented_incremental_train(self, data_manager):
        result = original_incremental_train(self, data_manager)
        _h029_record(self, data_manager)
        return result

    iCaRL.incremental_train = instrumented_incremental_train
    _H029_PATCH_INSTALLED = True


def _h032_old_class_mask(targets, known_classes):
    """Return exactly the old-class replay rows eligible for H032 POD."""
    import torch

    if not torch.is_tensor(targets):
        raise TypeError("H032 targets must be a torch.Tensor")
    if targets.ndim != 1:
        raise ValueError("H032 targets must be a rank-1 tensor")
    if not isinstance(known_classes, int) or known_classes < 0:
        raise ValueError("H032 known_classes must be a non-negative integer")
    return targets < known_classes


def _h032_pod_spatial_loss(current_fmap, old_fmap):
    """Deterministic normalized spatial-POD distance for Layer4 only.

    This is the same loss family as H030: square activations, sum them along
    each spatial axis, concatenate the two summaries, L2-normalize each
    sample, then average the samplewise L2 distances.  Detaching the old map
    inside the helper keeps its semantics safe when tested or reused outside
    the no-grad teacher forward.
    """
    import torch
    from torch.nn import functional as F

    if not torch.is_tensor(current_fmap) or not torch.is_tensor(old_fmap):
        raise TypeError("H032 feature maps must be torch tensors")
    if current_fmap.ndim != 4 or old_fmap.ndim != 4:
        raise ValueError("H032 spatial POD expects rank-4 [N,C,H,W] maps")
    if current_fmap.shape != old_fmap.shape:
        raise RuntimeError(
            "H032 Layer4 feature-map shape mismatch: {} vs {}".format(
                tuple(current_fmap.shape), tuple(old_fmap.shape)
            )
        )
    if current_fmap.shape[0] == 0:
        raise ValueError("H032 spatial POD cannot process an empty old-class batch")

    current_power = current_fmap.pow(2)
    old_power = old_fmap.detach().pow(2)
    current_descriptor = torch.cat(
        (
            current_power.sum(dim=3).flatten(1),
            current_power.sum(dim=2).flatten(1),
        ),
        dim=1,
    )
    old_descriptor = torch.cat(
        (
            old_power.sum(dim=3).flatten(1),
            old_power.sum(dim=2).flatten(1),
        ),
        dim=1,
    )
    current_descriptor = F.normalize(current_descriptor, p=2, dim=1)
    old_descriptor = F.normalize(old_descriptor, p=2, dim=1)
    return torch.linalg.vector_norm(current_descriptor - old_descriptor, dim=1).mean()


def _h032_synthetic_cpu_test():
    """Exercise the mask, POD value, and gradient boundary without RNG/GPU."""
    import torch

    targets = torch.tensor([0, 49, 50, 59], dtype=torch.long)
    mask = _h032_old_class_mask(targets, 50)
    if mask.tolist() != [True, True, False, False]:
        raise AssertionError("H032 old-class mask admitted a new-class target")

    old = (
        torch.arange(2 * 4 * 2 * 2, dtype=torch.float32)
        .reshape(2, 4, 2, 2)
        .requires_grad_()
    )
    identical = old.detach().clone().requires_grad_()
    zero = _h032_pod_spatial_loss(identical, old)
    if float(zero.item()) != 0.0:
        raise AssertionError("H032 identical Layer4 maps must have zero POD loss")

    changed = old.detach().clone()
    changed[1, 0, 0, 0] += 1.0
    changed.requires_grad_()
    positive = _h032_pod_spatial_loss(changed, old)
    if not bool(torch.isfinite(positive)) or positive.item() <= 0.0:
        raise AssertionError("H032 changed Layer4 map must have positive finite loss")
    positive.backward()
    if changed.grad is None or not bool(torch.isfinite(changed.grad).all()):
        raise AssertionError("H032 current Layer4 map did not receive finite gradients")
    if old.grad is not None:
        raise AssertionError("H032 old-network target received a gradient")
    return {
        "mask": mask.tolist(),
        "zero_pod": float(zero.item()),
        "positive_pod": float(positive.item()),
        "current_gradients": True,
        "old_gradients": False,
        "pod_weight": _H032_SPATIAL_COEFFICIENT,
    }


def _h032_static_contract_check():
    """Fail fast if the update hook drifts beyond Layer4 feature maps."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(_h032_update_representation))
    fmap_indices = {"fmaps": set(), "old_fmaps": set()}
    feature_accesses = []
    called_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        if not isinstance(node, ast.Subscript):
            continue
        index = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
        if isinstance(node.value, ast.Name) and node.value.id in fmap_indices:
            if not isinstance(index, ast.Constant) or not isinstance(index.value, int):
                raise AssertionError("H032 feature-map index must be a static integer")
            fmap_indices[node.value.id].add(index.value)
        if isinstance(index, ast.Constant) and index.value == "features":
            feature_accesses.append(node)
    if fmap_indices != {"fmaps": {3}, "old_fmaps": {3}}:
        raise AssertionError("H032 must access only Layer4 (fmaps[3])")
    if feature_accesses:
        raise AssertionError("H032 must not access or constrain the final embedding")
    required_calls = {"_h032_old_class_mask", "_h032_pod_spatial_loss"}
    if not required_calls.issubset(called_names):
        raise AssertionError("H032 update is missing its mask or Layer4 POD helper")
    return {"feature_map_indices": [3], "final_embedding_accesses": 0}


def _h032_update_representation(self, train_loader, test_loader, optimizer, scheduler):
    """H018 update plus old-class-only, Layer4-only spatial POD."""
    import logging
    import numpy as np
    import torch
    from torch.nn import functional as F
    from tqdm import tqdm
    from models import icarl as icarl_module
    from utils.toolkit import tensor2numpy

    if self._cur_task == 0:
        raise RuntimeError("H032 update hook reached Task 0")
    if self._old_network is None:
        raise RuntimeError("H032 requires a frozen old_network on incremental tasks")
    if any(parameter.requires_grad for parameter in self._old_network.parameters()):
        raise RuntimeError("H032 old_network must be frozen before distillation")

    diagnostic = {
        "task": int(self._cur_task),
        "known_classes": int(self._known_classes),
        "total_classes": int(self._total_classes),
        "epochs": int(icarl_module.epochs),
        "batches": 0,
        "samples": 0,
        "old_class_samples": 0,
        "new_class_samples": 0,
        "constrained_samples": 0,
        "new_class_constrained_samples": 0,
        "loss_sum": {"base": 0.0, "kd": 0.0, "pod": 0.0, "total": 0.0},
    }
    prog_bar = tqdm(range(icarl_module.epochs))
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
            loss_kd = icarl_module._KD_loss(
                logits[:, : self._known_classes],
                old_outputs["logits"],
                icarl_module.T,
            )

            old_mask = _h032_old_class_mask(targets, self._known_classes)
            old_count = int(old_mask.sum().item())
            new_count = int((~old_mask).sum().item())
            loss_pod = logits.new_zeros(())
            if old_count:
                fmaps = outputs.get("fmaps")
                old_fmaps = old_outputs.get("fmaps")
                if fmaps is None or len(fmaps) != 4:
                    raise RuntimeError("H032 expected four current ResNet feature maps")
                if old_fmaps is None or len(old_fmaps) != 4:
                    raise RuntimeError("H032 expected four old-network ResNet feature maps")
                loss_pod = _h032_pod_spatial_loss(
                    fmaps[3][old_mask], old_fmaps[3][old_mask]
                )

            loss = loss_base + loss_kd + _H032_SPATIAL_COEFFICIENT * loss_pod
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()

            diagnostic["batches"] += 1
            diagnostic["samples"] += int(targets.shape[0])
            diagnostic["old_class_samples"] += old_count
            diagnostic["new_class_samples"] += new_count
            diagnostic["constrained_samples"] += old_count
            diagnostic["loss_sum"]["base"] += float(loss_base.detach().item())
            diagnostic["loss_sum"]["kd"] += float(loss_kd.detach().item())
            diagnostic["loss_sum"]["pod"] += float(loss_pod.detach().item())
            diagnostic["loss_sum"]["total"] += float(loss.detach().item())

            _, preds = torch.max(logits, dim=1)
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)

        scheduler.step()
        train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
        mean_pod = diagnostic["loss_sum"]["pod"] / diagnostic["batches"]
        if epoch % 5 == 0:
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, "
                "Test_accy {:.2f}, H032_L4_POD {:.4f}"
            ).format(
                self._cur_task,
                epoch + 1,
                icarl_module.epochs,
                losses / len(train_loader),
                train_acc,
                test_acc,
                mean_pod,
            )
        else:
            info = (
                "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, "
                "H032_L4_POD {:.4f}"
            ).format(
                self._cur_task,
                epoch + 1,
                icarl_module.epochs,
                losses / len(train_loader),
                train_acc,
                mean_pod,
            )
        prog_bar.set_description(info)
    logging.info(info)

    if diagnostic["batches"] == 0:
        raise RuntimeError("H032 update encountered an empty train loader")
    if diagnostic["old_class_samples"] == 0:
        raise RuntimeError("H032 incremental training observed no replay rows")
    if diagnostic["constrained_samples"] != diagnostic["old_class_samples"]:
        raise AssertionError("H032 constrained-sample accounting drifted from replay rows")
    diagnostic["loss_mean"] = {
        name: value / diagnostic["batches"]
        for name, value in diagnostic.pop("loss_sum").items()
    }
    if not hasattr(self, "_h032_diagnostics"):
        self._h032_diagnostics = []
    self._h032_diagnostics.append(diagnostic)


def _h032_emit_diagnostic(self, data_manager):
    """Emit H032 engagement evidence after H029 records the final task."""
    if self._cur_task != data_manager.nb_tasks - 1:
        return
    measurements = getattr(self, "_h029_measurements", [])
    task5_layer4_cosine = None
    if measurements:
        final_layers = measurements[-1].get("layers", {})
        task5_layer4_cosine = final_layers.get("layer4", {}).get("cosine_similarity")
    artifact = {
        "schema": "openresearch.h032-replay-pod-l4.v1",
        "hypothesis": "old-class replay benefits from Layer4-only spatial POD without constraining new samples",
        "originalCommit": _H032_PARENT_COMMIT,
        "protocol": {
            "model": "iCaRL",
            "convnet": "resnet18",
            "memory_size": 6000,
            "seed": 1993,
            "controls": "H018 exact; H029 instrumentation preserved",
        },
        "intervention": {
            "layer": "layer4",
            "loss": "squared-activation height/width pooled spatial POD with L2 normalization",
            "coefficient": _H032_SPATIAL_COEFFICIENT,
            "target_policy": "targets < known_classes",
            "old_network_detached": True,
            "new_samples_constrained": False,
            "final_embedding_constrained": False,
        },
        "preregistration": {
            "target_engagement": {
                "label": "Task5",
                "task": 5,
                "metric": "H029 layer4 cosine_similarity",
                "baseline": 0.6813261509,
                "margin": 0.02,
                "threshold_formula": "0.6813261509 + 0.02",
                "threshold": 0.7013261509,
                "observed": task5_layer4_cosine,
            },
            "capability_support": {
                "metric": "official NME",
                "official_nme_threshold": 0.7132166667,
                "threshold": 0.7132166667,
            },
            "interpretation": "engagement-only is not repair support",
        },
        "diagnostics": getattr(self, "_h032_diagnostics", []),
        "synthetic_cpu_check": _h032_synthetic_cpu_test(),
        "static_contract_check": _h032_static_contract_check(),
    }
    print("H032_DIAGNOSTIC_JSON " + json.dumps(artifact, sort_keys=True, separators=(",", ":")), flush=True)


def _install_h032_distillation():
    """Install H032 and validate its bounded Layer4 contract once."""
    global _H032_PATCH_INSTALLED, _H032_CONTRACT_CHECKED
    if _H032_PATCH_INSTALLED:
        return
    if not _H032_CONTRACT_CHECKED:
        _h032_static_contract_check()
        _h032_synthetic_cpu_test()
        _H032_CONTRACT_CHECKED = True
    from models.icarl import iCaRL

    iCaRL._update_representation = _h032_update_representation
    _H032_PATCH_INSTALLED = True


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
    # H032's isolated incremental update and the H029 observer at that boundary
    # while leaving every H018 configuration value unchanged.
    _install_h032_distillation()
    _install_h029_instrumentation()
    return config
