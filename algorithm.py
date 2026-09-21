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

This file contains the epoch and hyperparameter configuration for iCaRL.
The actual iCaRL implementation is in the PyCIL repository (models/icarl.py).

Agents should modify the hyperparameters below, or replace the model_name
with a different continual learning algorithm from PyCIL (e.g., 'der', 'foster',
'memo', 'ewc', 'lwf', etc.).

To make deeper changes, agents can also modify models/icarl.py directly,
but must keep the BaseLearner interface (incremental_train, eval_task, after_task).
"""

import hashlib
import json
import pickle
import random


_H029_ORIGINAL_COMMIT = "7f187e7364c0e1a4f825c7c9c47061478e3c249b"
_H029_PATCH_INSTALLED = False


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


# ---------------------------------------------------------------------------
# H033: measurement-only NME geometry telemetry.
#
# This observer is deliberately kept in the adapter rather than in PyCIL.  It
# runs after iCaRL has built its rehearsal memory and exemplar means, and only
# reads the resulting model/data state.  In particular, the observer never
# changes an optimizer, loss, replay selection, loader order, or model tensor.
# ---------------------------------------------------------------------------

_H033_PATCH_INSTALLED = False


def _h033_bytes_digest(value):
    """Stable digest for an RNG state or a tensor/array snapshot."""
    return hashlib.sha256(pickle.dumps(value, protocol=4)).hexdigest()


def _h033_rng_snapshot():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def _h033_restore_rng(snapshot):
    import numpy as np
    import torch

    random.setstate(snapshot["python"])
    np.random.set_state(snapshot["numpy"])
    torch.set_rng_state(snapshot["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def _h033_rng_equal(left, right):
    import torch

    if _h033_bytes_digest(left["python"]) != _h033_bytes_digest(right["python"]):
        return False
    if _h033_bytes_digest(left["numpy"]) != _h033_bytes_digest(right["numpy"]):
        return False
    if not torch.equal(left["torch_cpu"], right["torch_cpu"]):
        return False
    if len(left["torch_cuda"]) != len(right["torch_cuda"]):
        return False
    return all(torch.equal(a, b) for a, b in zip(left["torch_cuda"], right["torch_cuda"]))


def _h033_array_digest(value):
    import numpy as np

    array = np.asarray(value)
    return hashlib.sha256(
        (str(array.dtype) + repr(tuple(array.shape))).encode("utf-8") + array.tobytes()
    ).hexdigest()


def _h033_network_digest(network):
    import torch
    from torch import nn

    if isinstance(network, nn.DataParallel):
        network = network.module
    digest = hashlib.sha256()
    for name, tensor in sorted(network.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    # Keep an explicit fc checksum in addition to state_dict: this is an
    # invariance contract for the classifier head even if a custom network
    # excludes a parameter from state_dict.
    for name in ("weight", "bias"):
        parameter = getattr(getattr(network, "fc", None), name, None)
        if parameter is not None:
            value = parameter.detach().cpu().contiguous()
            digest.update(("fc." + name).encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(repr(tuple(value.shape)).encode("utf-8"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _h033_fc_digest(network):
    import hashlib
    from torch import nn

    if isinstance(network, nn.DataParallel):
        network = network.module
    digest = hashlib.sha256()
    fc = getattr(network, "fc", None)
    if fc is None:
        return digest.hexdigest()
    for name in ("weight", "bias"):
        parameter = getattr(fc, name, None)
        if parameter is None:
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _h033_state_digests(learner):
    return {
        "network_state_dict": _h033_network_digest(learner._network),
        "fc_parameters": _h033_fc_digest(learner._network),
        "class_means": _h033_array_digest(learner._class_means),
        "data_memory": _h033_array_digest(learner._data_memory),
        "targets_memory": _h033_array_digest(learner._targets_memory),
    }


def _h033_percentiles(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise RuntimeError("H033 cannot summarize an empty sample group")
    if not np.isfinite(values).all():
        raise FloatingPointError("H033 encountered a non-finite metric")
    # Explicit linear interpolation makes the percentile convention portable
    # across NumPy versions (the old `interpolation` spelling is avoided).
    return {
        str(p): float(np.percentile(values, p, method="linear"))
        for p in (5, 25, 50, 75, 95)
    }


def _h033_sample_summary(mask, true_distance, impostor_distance, margin, error):
    import numpy as np

    if not np.any(mask):
        return {"count": 0}
    selected_margin = margin[mask]
    selected_error = error[mask]
    return {
        "count": int(mask.sum()),
        "error_rate": float(np.mean(selected_error)),
        "accuracy": float(1.0 - np.mean(selected_error)),
        "true_distance_mean": float(np.mean(true_distance[mask])),
        "impostor_distance_mean": float(np.mean(impostor_distance[mask])),
        "margin_mean": float(np.mean(selected_margin)),
        "true_distance_percentiles": _h033_percentiles(true_distance[mask]),
        "impostor_distance_percentiles": _h033_percentiles(impostor_distance[mask]),
        "margin_percentiles": _h033_percentiles(selected_margin),
    }


def _h033_extract_normalized(learner, loader):
    import numpy as np
    # Use the learner's own extraction path so this observer cannot silently
    # diverge from BaseLearner._eval_nme (DataParallel handling, dtype, and
    # loader traversal all remain in one source of truth).
    from models.base import EPSILON

    vectors, labels = learner._extract_vectors(loader)
    vectors = np.asarray(vectors)
    labels = np.asarray(labels, dtype=np.int64)
    if vectors.ndim != 2 or labels.ndim != 1 or len(vectors) != len(labels):
        raise RuntimeError("H033 extracted vectors/labels have incompatible shapes")
    if len(vectors) == 0:
        raise RuntimeError("H033 received an empty deterministic loader")
    if not np.isfinite(vectors).all():
        raise FloatingPointError("H033 encountered non-finite embedding values")
    # This is intentionally the exact NumPy expression used by
    # BaseLearner._eval_nme, including its repository EPSILON value.
    vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    if not np.isfinite(vectors).all():
        raise FloatingPointError("H033 normalized embeddings are non-finite")
    return vectors, labels


def _h033_distance_metrics(vectors, labels, centers):
    import numpy as np
    from scipy.spatial.distance import cdist

    if centers.ndim != 2 or vectors.ndim != 2 or vectors.shape[1] != centers.shape[1]:
        raise RuntimeError("H033 vector/center shapes are incompatible")
    if labels.ndim != 1 or len(labels) != len(vectors):
        raise RuntimeError("H033 labels do not match extracted vectors")
    if np.any(labels < 0) or np.any(labels >= len(centers)):
        raise RuntimeError("H033 labels fall outside the class-center range")

    # BaseLearner._eval_nme calls scipy.spatial.distance.cdist and then
    # np.argsort.  Reuse both operations here so tie ordering and floating
    # point arithmetic are directly compatible with the official evaluator.
    distances = cdist(vectors, centers, "sqeuclidean")
    rows = np.arange(len(labels))
    true_distance = distances[rows, labels]
    impostor_distances = distances.copy()
    impostor_distances[rows, labels] = np.inf
    impostor_distance = impostor_distances.min(axis=1)
    predicted = np.argsort(distances, axis=1)[:, 0].astype(np.int64, copy=False)
    margin = impostor_distance - true_distance
    if not all(np.isfinite(a).all() for a in (true_distance, impostor_distance, margin)):
        raise FloatingPointError("H033 distance metrics are non-finite")
    return {
        "true_distance": true_distance,
        "impostor_distance": impostor_distance,
        "margin": margin,
        "predicted": predicted,
    }


def _h033_full_centers(learner, data_manager, total_classes, feature_dim):
    import numpy as np
    from torch.utils.data import DataLoader
    from models.base import EPSILON

    dataset = data_manager.get_dataset(
        np.arange(total_classes), source="train", mode="test"
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    # Keep full-train geometry on the same extraction and NumPy normalization
    # path as official NME.  The resulting matrix is diagnostic-only and is
    # released once sufficient statistics have been accumulated below.
    vectors, labels = learner._extract_vectors(loader)
    vectors = np.asarray(vectors)
    labels = np.asarray(labels, dtype=np.int64)
    if vectors.ndim != 2 or vectors.shape[1] != feature_dim:
        raise RuntimeError("H033 full-train vectors have an unexpected shape")
    if labels.ndim != 1 or len(vectors) != len(labels):
        raise RuntimeError("H033 full-train vectors/labels have incompatible shapes")
    if len(vectors) == 0:
        raise RuntimeError("H033 full-train loader yielded no samples")
    if np.any(labels < 0) or np.any(labels >= total_classes):
        raise RuntimeError("H033 full-train labels fall outside the class range")
    if not np.isfinite(vectors).all():
        raise FloatingPointError("H033 full-train embeddings are non-finite")
    vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    if not np.isfinite(vectors).all():
        raise FloatingPointError("H033 normalized full-train embeddings are non-finite")

    sums = np.zeros((total_classes, feature_dim), dtype=np.float64)
    squared_norm_sums = np.zeros(total_classes, dtype=np.float64)
    counts = np.zeros(total_classes, dtype=np.int64)
    vectors64 = vectors.astype(np.float64, copy=False)
    np.add.at(sums, labels, vectors64)
    np.add.at(squared_norm_sums, labels, np.sum(vectors64 * vectors64, axis=1))
    np.add.at(counts, labels, 1)
    if np.any(counts == 0):
        raise RuntimeError("H033 full-train center missing a seen class")
    means = sums / counts[:, None]
    if not np.isfinite(means).all():
        raise FloatingPointError("H033 full-train center is non-finite")
    # Normalize centers with the same NumPy/EPSILON convention as embeddings.
    means = (means.T / (np.linalg.norm(means.T, axis=0) + EPSILON)).T
    if not np.isfinite(means).all():
        raise FloatingPointError("H033 normalized full-train center is non-finite")
    dispersion = (
        squared_norm_sums / counts
        - 2.0 * np.sum(means * sums, axis=1) / counts
        + np.sum(means * means, axis=1)
    )
    dispersion = np.maximum(dispersion, 0.0)
    if not np.isfinite(means).all() or not np.isfinite(dispersion).all():
        raise FloatingPointError("H033 full centers/dispersion are non-finite")
    return means, dispersion


# ---------------------------------------------------------------------------
# H034: measurement-only center interpolation and confusion-flow telemetry.
#
# H034 deliberately consumes the arrays already extracted by H033.  It never
# invokes the network, data manager, evaluator, optimizer, or replay path, so
# the added measurements cannot introduce another loader traversal or gradient
# path.  Full-train centers remain diagnostic oracle information throughout.
# ---------------------------------------------------------------------------

_H034_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _h034_interpolate_centers(official_centers, full_centers, alpha, known_classes, old_only):
    import numpy as np
    from models.base import EPSILON

    official_centers = np.asarray(official_centers, dtype=np.float64)
    full_centers = np.asarray(full_centers, dtype=np.float64)
    if official_centers.ndim != 2 or official_centers.shape != full_centers.shape:
        raise RuntimeError("H034 official/full center shapes are incompatible")
    if not np.isfinite(official_centers).all() or not np.isfinite(full_centers).all():
        raise FloatingPointError("H034 received non-finite centers")
    if alpha not in _H034_ALPHAS:
        raise ValueError("H034 alpha is outside the pre-registered grid: {}".format(alpha))
    if not 0 <= known_classes <= len(official_centers):
        raise RuntimeError("H034 known_classes is outside the center range")

    interpolated = official_centers.copy()
    stop = known_classes if old_only else len(interpolated)
    interpolated[:stop] = (
        (1.0 - alpha) * official_centers[:stop] + alpha * full_centers[:stop]
    )
    # Use the repository convention requested by the protocol, including its
    # additive EPSILON.  At alpha=0 this may change center magnitudes by a few
    # ulps, so exact prediction equality is asserted separately below.
    interpolated = (
        interpolated.T
        / (np.linalg.norm(interpolated.T, axis=0) + EPSILON)
    ).T
    if not np.isfinite(interpolated).all():
        raise FloatingPointError("H034 interpolated centers are non-finite")
    return interpolated


def _h034_task_age_ids(class_ids, increments, total_classes):
    import numpy as np

    class_ids = np.asarray(class_ids, dtype=np.int64)
    increments = np.asarray(increments, dtype=np.int64)
    if increments.ndim != 1 or len(increments) == 0 or np.any(increments <= 0):
        raise RuntimeError("H034 task increments are invalid")
    boundaries = np.cumsum(increments)
    if int(boundaries[-1]) != int(total_classes):
        raise RuntimeError("H034 task increments do not cover all seen classes")
    if np.any(class_ids < 0) or np.any(class_ids >= total_classes):
        raise RuntimeError("H034 class id falls outside the seen-class range")
    return np.searchsorted(boundaries, class_ids, side="right").astype(np.int64)


def _h034_transition_counts(labels, official_predicted, interpolated_predicted, increments):
    import numpy as np

    labels = np.asarray(labels, dtype=np.int64)
    official_predicted = np.asarray(official_predicted, dtype=np.int64)
    interpolated_predicted = np.asarray(interpolated_predicted, dtype=np.int64)
    if labels.ndim != 1 or official_predicted.shape != labels.shape or interpolated_predicted.shape != labels.shape:
        raise RuntimeError("H034 transition arrays have incompatible shapes")
    if len(labels) == 0:
        raise RuntimeError("H034 cannot summarize empty transition arrays")

    total_classes = int(np.sum(increments))
    true_age = _h034_task_age_ids(labels, increments, total_classes)
    official_age = _h034_task_age_ids(official_predicted, increments, total_classes)
    interpolated_age = _h034_task_age_ids(interpolated_predicted, increments, total_classes)
    official_correct = official_predicted == labels
    interpolated_correct = interpolated_predicted == labels
    corrected_mask = ~official_correct & interpolated_correct
    harmed_mask = official_correct & ~interpolated_correct
    both_correct_mask = official_correct & interpolated_correct
    both_wrong_mask = ~official_correct & ~interpolated_correct
    changed_mask = official_predicted != interpolated_predicted

    by_true_age = {}
    flow_sample_total = 0
    flow_corrected_total = 0
    flow_harmed_total = 0
    for age in range(len(increments)):
        age_mask = true_age == age
        flows = []
        for source_age in range(len(increments)):
            for target_age in range(len(increments)):
                flow_mask = age_mask & (official_age == source_age) & (interpolated_age == target_age)
                count = int(np.sum(flow_mask))
                if count == 0:
                    continue
                corrected = int(np.sum(flow_mask & corrected_mask))
                harmed = int(np.sum(flow_mask & harmed_mask))
                flows.append(
                    {
                        "official_predicted_task_age": int(source_age),
                        "interpolated_predicted_task_age": int(target_age),
                        "sample_count": count,
                        "corrected": corrected,
                        "harmed": harmed,
                    }
                )
                flow_sample_total += count
                flow_corrected_total += corrected
                flow_harmed_total += harmed
        age_count = int(np.sum(age_mask))
        if sum(flow["sample_count"] for flow in flows) != age_count:
            raise AssertionError("H034 task-age prediction flows do not conserve samples")
        by_true_age["task_{}".format(age)] = {
            "sample_count": age_count,
            "corrected": int(np.sum(age_mask & corrected_mask)),
            "harmed": int(np.sum(age_mask & harmed_mask)),
            "prediction_age_flows": flows,
        }

    totals = {
        "sample_count": int(len(labels)),
        "prediction_changed": int(np.sum(changed_mask)),
        "corrected": int(np.sum(corrected_mask)),
        "harmed": int(np.sum(harmed_mask)),
        "both_correct": int(np.sum(both_correct_mask)),
        "both_wrong": int(np.sum(both_wrong_mask)),
        "wrong_to_different_wrong": int(np.sum(both_wrong_mask & changed_mask)),
    }
    if totals["corrected"] + totals["harmed"] + totals["both_correct"] + totals["both_wrong"] != len(labels):
        raise AssertionError("H034 correctness transitions do not conserve samples")
    if flow_sample_total != len(labels):
        raise AssertionError("H034 prediction-age flows do not conserve all samples")
    if flow_corrected_total != totals["corrected"] or flow_harmed_total != totals["harmed"]:
        raise AssertionError("H034 prediction-age flow outcomes do not conserve transitions")
    return {"totals": totals, "by_true_class_task_age": by_true_age}


def _h034_measure_variant(
    test_vectors,
    test_labels,
    official_centers,
    full_centers,
    official_metrics,
    known_classes,
    increments,
    alpha,
    old_only,
    final_task,
):
    import numpy as np

    centers = _h034_interpolate_centers(
        official_centers, full_centers, alpha, known_classes, old_only
    )
    metrics = _h033_distance_metrics(test_vectors, test_labels, centers)
    error = metrics["predicted"] != test_labels
    if alpha == 0.0 and not np.array_equal(metrics["predicted"], official_metrics["predicted"]):
        raise AssertionError("H034 alpha=0 predictions disagree with official NME")

    old_mask = test_labels < known_classes
    groups = {
        "aggregate": np.ones(len(test_labels), dtype=bool),
        "old": old_mask,
        "new": ~old_mask,
    }
    measurement = {
        "alpha": float(alpha),
        "groups": {
            name: _h033_sample_summary(
                mask,
                metrics["true_distance"],
                metrics["impostor_distance"],
                metrics["margin"],
                error,
            )
            for name, mask in groups.items()
        },
        "task_age_summaries": {},
    }
    lower = 0
    for age, size in enumerate(increments):
        upper = lower + size
        mask = (test_labels >= lower) & (test_labels < upper)
        measurement["task_age_summaries"]["task_{}".format(age)] = {
            "class_range": [int(lower), int(upper)],
            "metrics": _h033_sample_summary(
                mask,
                metrics["true_distance"],
                metrics["impostor_distance"],
                metrics["margin"],
                error,
            ),
        }
        lower = upper
    if lower != len(official_centers):
        raise RuntimeError("H034 task-age ranges do not cover all centers")
    if final_task:
        measurement["final_task_transitions"] = _h034_transition_counts(
            test_labels, official_metrics["predicted"], metrics["predicted"], increments
        )
    return measurement


def _h034_synthetic_checks():
    """Focused CPU checks for interpolation semantics and measurement purity."""
    import numpy as np
    from models.base import EPSILON

    official = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    full = np.asarray([[0.0, 1.0], [1.0, 0.0], [-1.0, 1.0]], dtype=np.float64)
    official_before = official.copy()
    full_before = full.copy()
    # Construct fixed model-shaped sentinels directly: synthetic checks must
    # not consume any RNG stream merely by creating a state/gradient fixture.
    state = {
        "weight": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "bias": np.asarray([0.5, -0.5], dtype=np.float64),
    }
    gradients = {name: None for name in state}
    state_before = {name: value.copy() for name, value in state.items()}
    gradients_before = dict(gradients)

    alpha_zero = _h034_interpolate_centers(official, full, 0.0, 1, False)
    alpha_one = _h034_interpolate_centers(official, full, 1.0, 1, False)
    old_only = _h034_interpolate_centers(official, full, 1.0, 1, True)
    expected_zero = official / (np.linalg.norm(official, axis=1, keepdims=True) + EPSILON)
    expected_one = full / (np.linalg.norm(full, axis=1, keepdims=True) + EPSILON)
    expected_old_only = official.copy()
    expected_old_only[0] = full[0]
    expected_old_only /= np.linalg.norm(expected_old_only, axis=1, keepdims=True) + EPSILON
    if not np.array_equal(alpha_zero, expected_zero):
        raise AssertionError("H034 synthetic alpha=0 endpoint check failed")
    if not np.array_equal(alpha_one, expected_one):
        raise AssertionError("H034 synthetic alpha=1 endpoint check failed")
    if not np.array_equal(old_only, expected_old_only):
        raise AssertionError("H034 synthetic old-only masking check failed")
    expected_norms = np.linalg.norm(expected_one, axis=1)
    if not np.array_equal(np.linalg.norm(alpha_one, axis=1), expected_norms):
        raise AssertionError("H034 synthetic normalization check failed")

    vectors = np.asarray([[0.99, 0.01], [0.01, 0.99], [0.7, 0.7]], dtype=np.float64)
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    official_metrics = _h033_distance_metrics(vectors, labels, official)
    zero_metrics = _h033_distance_metrics(vectors, labels, alpha_zero)
    if not np.array_equal(zero_metrics["predicted"], official_metrics["predicted"]):
        raise AssertionError("H034 synthetic alpha=0 prediction check failed")

    transitions = _h034_transition_counts(
        np.asarray([0, 1, 2, 3]),
        np.asarray([1, 1, 0, 3]),
        np.asarray([0, 0, 2, 1]),
        [2, 2],
    )
    if transitions["totals"]["sample_count"] != 4:
        raise AssertionError("H034 synthetic transition conservation check failed")
    if sum(
        group["sample_count"]
        for group in transitions["by_true_class_task_age"].values()
    ) != 4:
        raise AssertionError("H034 synthetic task-age conservation check failed")

    if not np.array_equal(official, official_before) or not np.array_equal(full, full_before):
        raise AssertionError("H034 synthetic interpolation mutated its inputs")
    if any(gradients[name] is not gradients_before[name] for name in gradients):
        raise AssertionError("H034 synthetic check created or changed gradients")
    if any(not np.array_equal(state[name], value) for name, value in state_before.items()):
        raise AssertionError("H034 synthetic check changed model state")
    return {
        "alpha_endpoints": True,
        "old_only_masking": True,
        "normalization": True,
        "alpha_zero_predictions": True,
        "transition_count_conservation": True,
        "gradient_and_state_irrelevance": True,
    }


def _h034_record(
    learner,
    data_manager,
    test_vectors,
    test_labels,
    official_centers,
    official_metrics,
    full_centers,
):
    import numpy as np

    if not hasattr(learner, "_h034_measurements"):
        learner._h034_measurements = []
        learner._h034_invariance_checks = []
        learner._h034_synthetic_checks = _h034_synthetic_checks()
    expected_task = len(learner._h034_measurements)
    if learner._cur_task != expected_task:
        raise AssertionError(
            "H034 duplicate/missing/out-of-order task: expected {}, got {}".format(
                expected_task, learner._cur_task
            )
        )

    rng_before = _h033_rng_snapshot()
    state_before = _h033_state_digests(learner)
    mode_before = bool(learner._network.training)
    increments = [int(value) for value in data_manager._increments[: learner._cur_task + 1]]
    if sum(increments) != learner._total_classes:
        raise RuntimeError("H034 observed increments do not match total_classes")
    final_task = learner._cur_task == data_manager.nb_tasks - 1
    variants = {"global": [], "old_only": []}
    for variant_name, old_only in (("global", False), ("old_only", True)):
        for alpha in _H034_ALPHAS:
            variants[variant_name].append(
                _h034_measure_variant(
                    test_vectors,
                    test_labels,
                    official_centers,
                    full_centers,
                    official_metrics,
                    learner._known_classes,
                    increments,
                    alpha,
                    old_only,
                    final_task,
                )
            )

    record = {
        "task": int(learner._cur_task),
        "known_classes": int(learner._known_classes),
        "total_classes": int(learner._total_classes),
        "official_nme_accuracy": float(
            np.mean(official_metrics["predicted"] == test_labels)
        ),
        "variants": variants,
    }
    if not np.isfinite(record["official_nme_accuracy"]):
        raise FloatingPointError("H034 official NME accuracy is non-finite")
    learner._h034_measurements.append(record)

    rng_after = _h033_rng_snapshot()
    state_after = _h033_state_digests(learner)
    checks = {
        "python_numpy_torch_rng_unchanged": _h033_rng_equal(rng_before, rng_after),
        "network_mode_unchanged": mode_before == bool(learner._network.training),
        "network_state_dict_unchanged": state_before["network_state_dict"] == state_after["network_state_dict"],
        "fc_parameters_unchanged": state_before["fc_parameters"] == state_after["fc_parameters"],
        "class_means_unchanged": state_before["class_means"] == state_after["class_means"],
        "data_memory_unchanged": state_before["data_memory"] == state_after["data_memory"],
        "targets_memory_unchanged": state_before["targets_memory"] == state_after["targets_memory"],
    }
    if not all(checks.values()):
        raise AssertionError("H034 invariance check failed: {}".format(checks))
    learner._h034_invariance_checks.append(checks)

    if not final_task:
        return
    observed_tasks = [measurement["task"] for measurement in learner._h034_measurements]
    expected_tasks = list(range(data_manager.nb_tasks))
    if observed_tasks != expected_tasks:
        raise AssertionError(
            "H034 missing/duplicate tasks: expected {}, got {}".format(
                expected_tasks, observed_tasks
            )
        )

    final_record = learner._h034_measurements[-1]
    official_accuracy = final_record["official_nme_accuracy"]
    fixed_candidates = []
    exploratory_candidates = []
    for variant_name in ("global", "old_only"):
        for measurement in final_record["variants"][variant_name]:
            candidate = {
                "variant": variant_name,
                "alpha": measurement["alpha"],
                "accuracy": measurement["groups"]["aggregate"]["accuracy"],
            }
            candidate["accuracy_delta_pp"] = float(
                (candidate["accuracy"] - official_accuracy) * 100.0
            )
            exploratory_candidates.append(candidate)
            if candidate["alpha"] in (0.25, 0.5, 0.75):
                fixed_candidates.append(candidate)
    exploratory_best = max(
        exploratory_candidates,
        key=lambda candidate: (candidate["accuracy"], -candidate["alpha"], candidate["variant"] == "global"),
    )
    threshold_met = any(
        candidate["accuracy_delta_pp"] >= 0.5 for candidate in fixed_candidates
    )
    artifact = {
        "schema": "openresearch.h034-center-interpolation.v1",
        "hypothesis": "fixed partial movement from exemplar centers toward full-train centers exposes usable NME geometry headroom and task-age-asymmetric confusion flow",
        "protocol": {
            "model": "iCaRL",
            "convnet": "resnet18",
            "memory_size": 6000,
            "seed": 1993,
            "alpha_grid": list(_H034_ALPHAS),
            "global": "interpolate every seen class center",
            "old_only": "interpolate classes below known_classes; keep current-task centers official",
            "reuse": "H033 normalized official test embeddings, exemplar centers, and full-train centers; no additional extraction",
        },
        "metric_definitions": {
            "interpolation": "normalize((1-alpha)*official_center+alpha*full_train_center) using repository EPSILON",
            "squared_distance": "scipy cdist sqeuclidean, matching official NME ordering",
            "margin": "nearest impostor distance - true-class distance; positive is correct",
            "prediction_age_flow": "counts indexed by true-class task age, official-predicted task age, and interpolated-predicted task age",
        },
        "preregistration": {
            "geometry_headroom_support": "any fixed partial-interpolation candidate (global or old_only; alpha 0.25, 0.5, or 0.75) reaches final-task official NME +0.5 percentage points",
            "pure_full_center_endpoint": "alpha=1.0 is an oracle endpoint and is excluded from the support gate",
            "exploratory_best_alpha": "descriptive and non-deployable because full-train centers are used",
        },
        "measurements": learner._h034_measurements,
        "invariance_checks": learner._h034_invariance_checks,
        "synthetic_checks": learner._h034_synthetic_checks,
        "preregistered_reading": {
            "official_final_task_accuracy": official_accuracy,
            "support_threshold_pp": 0.5,
            "fixed_partial_candidates": fixed_candidates,
            "geometry_headroom_supported": bool(threshold_met),
            "exploratory_best": exploratory_best,
        },
        "limitations": [
            "every alpha above zero uses full-train centers and is non-deployable diagnostic telemetry",
            "the best alpha is selected retrospectively and is descriptive, not a validated policy",
            "center interpolation does not change training and cannot establish a causal repair",
            "transition flows retain counts and summaries rather than per-sample records",
        ],
    }
    print("H034_CENTER_INTERPOLATION_JSON " + json.dumps(artifact, sort_keys=True, separators=(",", ":")), flush=True)


def _h033_record(learner, data_manager):
    import numpy as np

    if learner.args.get("convnet_type", "").lower() != "resnet18":
        raise ValueError("H033 requires the H018 ResNet-18 protocol")
    if learner.args.get("memory_size") != 6000 or learner.args.get("seed") != 1993:
        raise ValueError("H033 requires memory_size=6000 and seed=1993")
    if learner._cur_task < 0 or learner._total_classes <= 0:
        raise RuntimeError("H033 cannot measure an uninitialized task")

    if not hasattr(learner, "_h033_measurements"):
        learner._h033_measurements = []
        learner._h033_invariance_checks = []
    rng_before = _h033_rng_snapshot()
    state_before = _h033_state_digests(learner)
    was_training = learner._network.training
    record = None
    try:
        test_vectors, test_labels = _h033_extract_normalized(learner, learner.test_loader)
        class_means = np.asarray(learner._class_means, dtype=np.float64)
        if class_means.shape != (learner._total_classes, test_vectors.shape[1]):
            raise RuntimeError("H033 _class_means shape does not match final embeddings")
        if not np.isfinite(class_means).all():
            raise FloatingPointError("H033 _class_means are non-finite")

        official = _h033_distance_metrics(test_vectors, test_labels, class_means)
        # Re-run the evaluator's own path and require exact top-1 agreement,
        # including scipy's deterministic tie ordering.
        evaluator_pred, evaluator_targets = learner._eval_nme(
            learner.test_loader, learner._class_means
        )
        if not np.array_equal(test_labels, evaluator_targets):
            raise AssertionError("H033 test-label order differs from official NME loader")
        if not np.array_equal(official["predicted"], evaluator_pred[:, 0]):
            raise AssertionError("H033 failed exact official-NME reproduction")

        full_centers, dispersion = _h033_full_centers(
            learner, data_manager, learner._total_classes, test_vectors.shape[1]
        )
        _h034_record(
            learner,
            data_manager,
            test_vectors,
            test_labels,
            class_means,
            official,
            full_centers,
        )
        full = _h033_distance_metrics(test_vectors, test_labels, full_centers)
        old_mask = test_labels < learner._known_classes
        new_mask = ~old_mask
        official_error = official["predicted"] != test_labels
        full_error = full["predicted"] != test_labels
        groups = {
            "total": np.ones(len(test_labels), dtype=bool),
            "old": old_mask,
            "new": new_mask,
        }
        sample_groups = {
            "official": {
                name: _h033_sample_summary(
                    mask,
                    official["true_distance"],
                    official["impostor_distance"],
                    official["margin"],
                    official_error,
                )
                for name, mask in groups.items()
            },
            "full_center": {
                name: _h033_sample_summary(
                    mask,
                    full["true_distance"],
                    full["impostor_distance"],
                    full["margin"],
                    full_error,
                )
                for name, mask in groups.items()
            },
        }

        # Task-age summaries use mapped class ranges, which are stable under
        # the H018 schedule and avoid retaining per-sample arrays.
        task_ages = {}
        lower = 0
        for age, size in enumerate(data_manager._increments[: learner._cur_task + 1]):
            upper = lower + size
            mask = (test_labels >= lower) & (test_labels < upper)
            task_ages["task_{}".format(age)] = {
                "class_range": [int(lower), int(upper)],
                "official": _h033_sample_summary(
                    mask,
                    official["true_distance"],
                    official["impostor_distance"],
                    official["margin"],
                    official_error,
                ),
                "full_center": _h033_sample_summary(
                    mask,
                    full["true_distance"],
                    full["impostor_distance"],
                    full["margin"],
                    full_error,
                ),
            }
            lower = upper

        per_class = []
        for class_id in range(learner._total_classes):
            class_mask = test_labels == class_id
            corrected = int(np.sum(class_mask & official_error & ~full_error))
            harmed = int(np.sum(class_mask & ~official_error & full_error))
            ex = class_means[class_id]
            center = full_centers[class_id]
            class_official = _h033_sample_summary(
                class_mask,
                official["true_distance"],
                official["impostor_distance"],
                official["margin"],
                official_error,
            )
            class_full = _h033_sample_summary(
                class_mask,
                full["true_distance"],
                full["impostor_distance"],
                full["margin"],
                full_error,
            )
            per_class.append(
                {
                    "class": int(class_id),
                    "test_count": int(class_mask.sum()),
                    "exemplar_to_full_squared_distance": float(np.sum((ex - center) ** 2)),
                    "exemplar_to_full_cosine": float(np.dot(ex, center) / ((np.linalg.norm(ex) * np.linalg.norm(center)) + 1e-8)),
                    "full_class_dispersion": float(dispersion[class_id]),
                    "official_errors": int(np.sum(class_mask & official_error)),
                    "full_errors": int(np.sum(class_mask & full_error)),
                    "corrected_by_full_center": corrected,
                    "harmed_by_full_center": harmed,
                    "correction_fraction": float(corrected / max(1, int(np.sum(class_mask & official_error)))),
                    "harm_fraction": float(harmed / max(1, int(np.sum(class_mask & ~official_error)))),
                    "official_margin_percentiles": class_official.get("margin_percentiles", {}),
                    "full_center_margin_percentiles": class_full.get("margin_percentiles", {}),
                }
            )
        official_accuracy = float(np.mean(~official_error))
        full_accuracy = float(np.mean(~full_error))
        official_errors = int(np.sum(official_error))
        corrected = int(np.sum(official_error & ~full_error))
        harmed = int(np.sum(~official_error & full_error))
        record = {
            "task": int(learner._cur_task),
            "known_classes": int(learner._known_classes),
            "total_classes": int(learner._total_classes),
            "official_nme": {
                "accuracy": official_accuracy,
                "error_count": official_errors,
                "groups": sample_groups["official"],
            },
            "full_center_nme": {
                "accuracy": full_accuracy,
                "groups": sample_groups["full_center"],
            },
            "comparison": {
                "accuracy_delta_pp": float((full_accuracy - official_accuracy) * 100.0),
                "official_errors_corrected": corrected,
                "official_correct_predictions_harmed": harmed,
                "net_correction_fraction_of_official_errors": float((corrected - harmed) / max(1, official_errors)),
            },
            "task_age_summaries": task_ages,
            "per_class": per_class,
        }
        if not np.isfinite(official_accuracy) or not np.isfinite(full_accuracy):
            raise FloatingPointError("H033 accuracy is non-finite")
        learner._h033_measurements.append(record)
    finally:
        if was_training:
            learner._network.train()
        else:
            learner._network.eval()
        _h033_restore_rng(rng_before)
        rng_after = _h033_rng_snapshot()
        state_after = _h033_state_digests(learner)
        checks = {
            "python_numpy_torch_rng_restored": _h033_rng_equal(rng_before, rng_after),
            "network_state_dict_unchanged": state_before["network_state_dict"] == state_after["network_state_dict"],
            "fc_parameters_unchanged": state_before["fc_parameters"] == state_after["fc_parameters"],
            "class_means_unchanged": state_before["class_means"] == state_after["class_means"],
            "data_memory_unchanged": state_before["data_memory"] == state_after["data_memory"],
            "targets_memory_unchanged": state_before["targets_memory"] == state_after["targets_memory"],
        }
        if not all(checks.values()):
            raise AssertionError("H033 invariance check failed: {}".format(checks))
        learner._h033_invariance_checks.append(checks)

    if learner._cur_task == data_manager.nb_tasks - 1:
        if len(learner._h033_measurements) != data_manager.nb_tasks:
            raise AssertionError("H033 did not record exactly one measurement per task")
        artifact = {
            "schema": "openresearch.h033-nme-geometry.v1",
            "hypothesis": "official NME error is explained by exemplar-prototype bias versus final-embedding overlap/margin erosion",
            "protocol": {
                "model": "iCaRL",
                "convnet": "resnet18",
                "memory_size": 6000,
                "seed": 1993,
                "test_transform": "deterministic mode=test transform",
                "loaders": "non-shuffled; full-train center loader uses num_workers=0",
            },
            "metric_definitions": {
                "normalized_embedding": "extract_vector(x)/(||extract_vector(x)||_2+1e-8)",
                "squared_distance": "||normalized_test_embedding - normalized_class_center||_2^2",
                "margin": "nearest impostor distance - true-class distance; positive is correct",
                "percentiles": "deterministic linear np.percentile at 5,25,50,75,95",
                "dispersion": "mean squared distance of normalized train embeddings to normalized full-train center",
            },
            "preregistration": {
                "prototype_bias_support": "final-task full-center NME improves official NME by >=0.5 percentage points OR net correction fraction of official errors >=0.20",
                "otherwise": "representation/decision-margin overlap remains the dominant measured bottleneck",
                "interpretation": "diagnosis only; not a repair or causal proof",
            },
            "measurements": learner._h033_measurements,
            "invariance_checks": learner._h033_invariance_checks,
            "limitations": [
                "full-train centers are non-deployable diagnostic references",
                "center comparisons are descriptive and do not establish causal dominance",
                "percentiles retain summaries, not per-sample arrays",
                "class-conditional test summaries inherit the evaluator's mapped test split",
            ],
        }
        final_comparison = learner._h033_measurements[-1]["comparison"]
        artifact["preregistered_reading"] = {
            "final_task_accuracy_delta_pp": final_comparison["accuracy_delta_pp"],
            "final_task_net_correction_fraction": final_comparison[
                "net_correction_fraction_of_official_errors"
            ],
            "prototype_bias_threshold_met": bool(
                final_comparison["accuracy_delta_pp"] >= 0.5
                or final_comparison["net_correction_fraction_of_official_errors"] >= 0.20
            ),
        }
        print("H033_NME_GEOMETRY_JSON " + json.dumps(artifact, sort_keys=True, separators=(",", ":")), flush=True)


def _install_h033_instrumentation():
    global _H033_PATCH_INSTALLED
    if _H033_PATCH_INSTALLED:
        return
    from models.icarl import iCaRL

    original_incremental_train = iCaRL.incremental_train

    def instrumented_incremental_train(self, data_manager):
        result = original_incremental_train(self, data_manager)
        _h033_record(self, data_manager)
        return result

    iCaRL.incremental_train = instrumented_incremental_train
    _H033_PATCH_INSTALLED = True


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
    _install_h033_instrumentation()
    return config
