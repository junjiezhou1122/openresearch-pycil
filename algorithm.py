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
H010: is the NME ceiling the DECISION RULE or the REPRESENTATION?
-------------------------------------------------------------------------------
Everything about training is identical to the verified H001 configuration
(iCaRL + ResNet-18, init_epoch=100, epochs=50, seed 1993, memory_size=2000).

Wave 1 (H008, H007b, H009c) established that the CNN-side residual is
classifier-norm / affine-vs-angular readout geometry and that the NME-side
residual is *not* prototype estimation, and it attributed the remaining NME
ceiling to "the frozen representation". That attribution rests on a NEGATIVE
result (better prototypes do not help). H010 tests it positively:

  The registered NME rule summarizes the stored exemplars of each class by
  their L2-normalized mean and predicts with an argmin over those means. On the
  SAME frozen features and the SAME stored exemplars, H010 fits a closed-form
  (ridge) linear readout over the raw exemplar features and predicts with it.

  * If the ridge readout beat the registered nearest-mean rule, the exemplar
    mean summary (i.e. the DECISION RULE) is what caps NME and the frozen
    representation still carries unused class information.
  * If it does not, the ceiling is the exemplars/representation and wave 1's
    attribution is confirmed by a positive control.

Scope / auditability:
  * The ridge readout is CLOSED FORM (no optimiser, no iteration, no gradient
    step) and its regularisation is fixed a priori as
    ``lam = 1e-3 * trace(X^T X) / d`` -- a single data-scaled value, never
    tuned on the test set.
  * ``self._class_means``, ``_data_memory``, ``_targets_memory``, ``fc.weight``,
    ``fc.bias``, the whole ``state_dict`` and the convnet are never written;
    sha256 checksums are compared before/after every ``_eval_nme`` call, printed
    per stage, and any mismatch raises (fail fast).
  * The exemplar features are extracted with exactly the call iCaRL itself uses
    to build the class means (``get_dataset([], source='train', mode='test',
    appendent=(_data_memory, _targets_memory))``), so NME and the ridge readout
    see the same information; the only difference is mean-summary + argmin
    versus a fitted linear map on the raw features.
  * Building/iterating that DataLoader draws ``_base_seed`` from the GLOBAL
    torch generator, so it is wrapped in a CPU **and** CUDA RNG
    snapshot/restore whose restoration is verified and printed per stage.
  * The registered NME rule is scored in the SAME pass and emitted as an
    in-run control, so the run proves the patch is a readout swap only.
  * The returned prediction is the H010 ridge readout.
"""

import hashlib
import json

import numpy as np
import torch

import models.base as base_module
from scipy.spatial.distance import cdist

_H010_PATCH_INSTALLED = False
_H010_EPS = base_module.EPSILON
# A priori, data-scaled ridge strength. Never tuned on the test split.
_H010_RIDGE_SCALE = 1e-3


def _h010_sha_array(value):
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _h010_sha_tensor(tensor):
    return hashlib.sha256(
        tensor.detach().to("cpu").contiguous().numpy().tobytes()
    ).hexdigest()


def _h010_sha_network(net):
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(_h010_sha_tensor(state[name]).encode("ascii"))
    return digest.hexdigest()


def _h010_jsonable_grouped(grouped):
    return {str(k): (float(v) if v is not None else None) for k, v in grouped.items()}


def _h010_eval_nme(self, loader, class_means):
    """H010: same features/exemplars, ridge linear readout vs nearest-mean."""
    import torch.nn as nn

    data_manager = getattr(self, "_h010_data_manager", None)
    nb_old = int(self._known_classes)
    nb_total = int(self._total_classes)
    topk = int(self.topk)

    net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

    persistent = class_means
    cm_before = _h010_sha_array(persistent)
    data_before = _h010_sha_array(self._data_memory)
    targets_before = _h010_sha_array(self._targets_memory)
    weight_before = _h010_sha_tensor(net.fc.weight)
    bias_before = None if net.fc.bias is None else _h010_sha_tensor(net.fc.bias)
    network_before = _h010_sha_network(self._network)

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    exemplar_features = None
    exemplar_targets = None
    n_exemplars = 0
    try:
        if data_manager is not None and len(self._data_memory) != 0:
            # Mirrors iCaRL's own exemplar-mean call exactly: without
            # ret_data=True get_dataset returns the dataset itself.
            ex_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(self._data_memory, self._targets_memory),
            )
            ex_loader = base_module.DataLoader(
                ex_dset,
                batch_size=base_module.batch_size,
                shuffle=False,
                num_workers=4,
            )
            exemplar_features, exemplar_targets = self._extract_vectors(ex_loader)
            n_exemplars = int(len(exemplar_targets))
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    rng_cpu_restored = bool(torch.get_rng_state().equal(cpu_rng))
    rng_cuda_restored = None
    if cuda_rng is not None:
        current_cuda = torch.cuda.get_rng_state_all()
        rng_cuda_restored = bool(
            len(current_cuda) == len(cuda_rng)
            and all(a.equal(b) for a, b in zip(current_cuda, cuda_rng))
        )

    # ---- score both rules on the same extracted test features ----
    self._network.eval()
    vectors_t, y_true = self._extract_vectors(loader)
    vectors_t = (vectors_t.T / (np.linalg.norm(vectors_t.T, axis=0) + _H010_EPS)).T

    # (a) registered NME rule (in-run control)
    scores_nme = cdist(persistent, vectors_t, "sqeuclidean").T
    preds_nme = np.argsort(scores_nme, axis=1)[:, :topk]

    # (b) H010 closed-form ridge linear readout on the raw exemplar features
    ridge_ok = False
    lam = None
    preds_ridge = preds_nme
    n_classes_used = None
    if exemplar_features is not None and n_exemplars > 0:
        X = exemplar_features
        X = (X.T / (np.linalg.norm(X.T, axis=0) + _H010_EPS)).T
        Y = np.zeros((X.shape[0], nb_total), dtype=np.float64)
        Y[np.arange(X.shape[0]), exemplar_targets.astype(int)] = 1.0
        gram = X.T @ X
        lam = float(_H010_RIDGE_SCALE * np.trace(gram) / gram.shape[0])
        W = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), X.T @ Y)
        logits = vectors_t @ W
        preds_ridge = np.argsort(-logits, axis=1)[:, :topk]
        ridge_ok = True
        n_classes_used = int(nb_total)

    cm_after = _h010_sha_array(persistent)
    data_after = _h010_sha_array(self._data_memory)
    targets_after = _h010_sha_array(self._targets_memory)
    weight_after = _h010_sha_tensor(net.fc.weight)
    bias_after = None if net.fc.bias is None else _h010_sha_tensor(net.fc.bias)
    network_after = _h010_sha_network(self._network)

    flags = {
        "class_means_sha_unchanged": cm_after == cm_before,
        "data_memory_sha_unchanged": data_after == data_before,
        "targets_memory_sha_unchanged": targets_after == targets_before,
        "fc_weight_sha_unchanged": weight_after == weight_before,
        "fc_bias_sha_unchanged": bias_after == bias_before,
        "network_sha_unchanged": network_after == network_before,
        "rng_cpu_restored": rng_cpu_restored,
        "rng_cuda_restored": rng_cuda_restored,
        "ridge_fitted": ridge_ok,
    }
    invariance_ok = all(v for v in flags.values() if v is not None)

    diag = {
        "stage": int(self._cur_task),
        "nb_old": nb_old,
        "nb_total": nb_total,
        "ridge_scale": _H010_RIDGE_SCALE,
        "ridge_lambda": lam,
        "n_exemplars": n_exemplars,
        "n_classes_used": n_classes_used,
        "nme_registered_grouped_in_run_control": _h010_jsonable_grouped(
            base_module.accuracy(preds_nme.T[0], y_true, nb_old)
        ),
        "ridge_grouped": _h010_jsonable_grouped(
            base_module.accuracy(preds_ridge.T[0], y_true, nb_old)
        ),
        "invariance_flags": flags,
        "invariance_ok": bool(invariance_ok),
    }
    print("H010_DIAG " + json.dumps(diag, sort_keys=True), flush=True)

    if not invariance_ok:
        raise RuntimeError(
            "H010 invariance violation (persistent state or RNG changed): "
            + json.dumps(diag, sort_keys=True)
        )

    return preds_ridge, y_true


def _install_h010_linear_probe():
    global _H010_PATCH_INSTALLED
    if _H010_PATCH_INSTALLED:
        return False

    from models.icarl import iCaRL

    original_incremental_train = iCaRL.incremental_train

    def _h010_incremental_train(self, data_manager):
        self._h010_data_manager = data_manager
        return original_incremental_train(self, data_manager)

    iCaRL.incremental_train = _h010_incremental_train
    base_module.BaseLearner._eval_nme = _h010_eval_nme
    _H010_PATCH_INSTALLED = True

    print(
        "H010_PATCH_INSTALLED "
        + json.dumps(
            {
                "target": "BaseLearner._eval_nme (inherited by iCaRL)",
                "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
                "mode": "evaluation-only; persistent state never written",
                "readout": "closed-form ridge on the raw exemplar features",
                "ridge_scale": _H010_RIDGE_SCALE,
                "training_untouched": True,
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
    # H010: evaluation-only swap of the decision RULE (nearest-mean -> closed-form
    # ridge linear readout) on the same frozen features and the same stored
    # exemplars. Training, memory, exemplar selection and class means untouched.
    _install_h010_linear_probe()
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
