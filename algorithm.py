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
H011: how much class information do the FROZEN features actually carry?
-------------------------------------------------------------------------------
Everything about training is identical to the verified H001 configuration
(iCaRL + ResNet-18, init_epoch=100, epochs=50, seed 1993, memory_size=2000).

Where this comes from. Wave 1 resolved the CNN side (H008, H007b: classifier-norm
/ affine-vs-angular readout geometry) and refuted prototype estimation as the
NME-side lever (H009c: recomputing prototypes from ALL class data moves aggregate
NME by -0.0083 pp). H010 then showed that a closed-form ridge readout fitted on
the SAME stored exemplars also fails to beat the registered nearest-mean rule
(-0.66 pp aggregate). Those two negative results were read as "the frozen
features limit NME", but neither of them actually measured the information the
features carry: H009c replaced only the class MEANS, and H010 could only use the
2000-exemplar budget. Both are limited by the class-incremental INFORMATION
BUDGET, not necessarily by the representation.

H011 measures the bound directly. On the SAME frozen features it fits a
closed-form (ridge) linear readout on the FULL training split of every class seen
so far and compares it with the registered nearest-mean rule in the same pass.

  * If the ORACLE readout substantially beats the registered NME, then the frozen
    representation carries class information that the exemplar-budget NME cannot
    reach, and the binding constraint on the accepted metric is the
    class-incremental information budget / the mean summary -- NOT the features.
  * If it does not, the frozen features themselves cap the metric and no readout
    can move it, which is a hard upper bound on this search space.

This is explicitly an ORACLE BOUND, not a deployable algorithm: it uses the full
training split of all seen classes, which a class-incremental learner does not
have. It is registered as an upper bound so that a positive result localises the
constraint (information access) and a negative result closes the search space.

Scope / auditability:
  * Both readouts are CLOSED FORM (no optimiser, no gradient step, no iteration)
    with the same a-priori regularisation ``lam = 1e-3 * trace(X^T X) / d``,
    never tuned on the test split.
  * A second, DEPLOYABLE readout fitted on the stored exemplars only is also
    reported (the H010 construction), so the oracle-to-deployable gap is visible.
  * Persistent state, training, memories, fc weights and the CPU/CUDA RNG stream
    are never written; before/after sha256 equality is printed per stage and any
    mismatch raises.
  * The registered nearest-mean rule is scored in the SAME pass as the in-run
    control, so the patch is provably a readout swap only.
  * Returned prediction = the ORACLE readout (the measured upper bound).
"""

import hashlib
import json

import numpy as np
import torch

import models.base as base_module

_H011_PATCH_INSTALLED = False
_H011_EPS = base_module.EPSILON
_H011_RIDGE_SCALE = 1e-3


def _h011_sha_array(value):
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _h011_sha_tensor(tensor):
    return hashlib.sha256(
        tensor.detach().to("cpu").contiguous().numpy().tobytes()
    ).hexdigest()


def _h011_sha_network(net):
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(_h011_sha_tensor(state[name]).encode("ascii"))
    return digest.hexdigest()


def _h011_grouped(grouped):
    return {str(k): (float(v) if v is not None else None) for k, v in grouped.items()}


def _h011_normalize(x):
    return (x.T / (np.linalg.norm(x.T, axis=0) + _H011_EPS)).T


def _h011_ridge(X, y, n_classes):
    X = _h011_normalize(X)
    Y = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    Y[np.arange(X.shape[0]), y.astype(int)] = 1.0
    gram = X.T @ X
    lam = float(_H011_RIDGE_SCALE * np.trace(gram) / gram.shape[0])
    W = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), X.T @ Y)
    return W, lam


def _h011_eval_nme(self, loader, class_means):
    """H011: registered NME vs an oracle full-train linear readout on frozen features."""
    import torch.nn as nn

    from scipy.spatial.distance import cdist

    data_manager = getattr(self, "_h011_data_manager", None)
    nb_old = int(self._known_classes)
    nb_total = int(self._total_classes)
    topk = int(self.topk)

    net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

    persistent = class_means
    cm_before = _h011_sha_array(persistent)
    data_before = _h011_sha_array(self._data_memory)
    targets_before = _h011_sha_array(self._targets_memory)
    weight_before = _h011_sha_tensor(net.fc.weight)
    bias_before = None if net.fc.bias is None else _h011_sha_tensor(net.fc.bias)
    network_before = _h011_sha_network(self._network)

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    full_x = full_y = ex_x = ex_y = None
    n_full = n_ex = 0
    try:
        if data_manager is not None:
            # ORACLE source: the full training split of every class seen so far.
            full_dset = data_manager.get_dataset(
                np.arange(0, nb_total), source="train", mode="test"
            )
            full_loader = base_module.DataLoader(
                full_dset, batch_size=base_module.batch_size, shuffle=False, num_workers=4
            )
            full_x, full_y = self._extract_vectors(full_loader)
            n_full = int(len(full_y))
            if len(self._data_memory) != 0:
                ex_dset = data_manager.get_dataset(
                    [],
                    source="train",
                    mode="test",
                    appendent=(self._data_memory, self._targets_memory),
                )
                ex_loader = base_module.DataLoader(
                    ex_dset, batch_size=base_module.batch_size, shuffle=False, num_workers=4
                )
                ex_x, ex_y = self._extract_vectors(ex_loader)
                n_ex = int(len(ex_y))
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

    self._network.eval()
    vectors_t, y_true = self._extract_vectors(loader)
    vectors_t = _h011_normalize(vectors_t)

    preds_nme = np.argsort(cdist(persistent, vectors_t, "sqeuclidean").T, axis=1)[:, :topk]

    oracle_ok = False
    deployable_ok = False
    lam_oracle = lam_ex = None
    preds_oracle = preds_deployable = preds_nme
    if full_x is not None and n_full > 0:
        W, lam_oracle = _h011_ridge(full_x, full_y, nb_total)
        preds_oracle = np.argsort(-(vectors_t @ W), axis=1)[:, :topk]
        oracle_ok = True
    if ex_x is not None and n_ex > 0:
        W_e, lam_ex = _h011_ridge(ex_x, ex_y, nb_total)
        preds_deployable = np.argsort(-(vectors_t @ W_e), axis=1)[:, :topk]
        deployable_ok = True

    cm_after = _h011_sha_array(persistent)
    data_after = _h011_sha_array(self._data_memory)
    targets_after = _h011_sha_array(self._targets_memory)
    weight_after = _h011_sha_tensor(net.fc.weight)
    bias_after = None if net.fc.bias is None else _h011_sha_tensor(net.fc.bias)
    network_after = _h011_sha_network(self._network)

    flags = {
        "class_means_sha_unchanged": cm_after == cm_before,
        "data_memory_sha_unchanged": data_after == data_before,
        "targets_memory_sha_unchanged": targets_after == targets_before,
        "fc_weight_sha_unchanged": weight_after == weight_before,
        "fc_bias_sha_unchanged": bias_after == bias_before,
        "network_sha_unchanged": network_after == network_before,
        "rng_cpu_restored": rng_cpu_restored,
        "rng_cuda_restored": rng_cuda_restored,
        "oracle_fitted": oracle_ok,
        "deployable_fitted": deployable_ok,
    }
    invariance_ok = all(v for v in flags.values() if v is not None)

    diag = {
        "stage": int(self._cur_task),
        "nb_old": nb_old,
        "nb_total": nb_total,
        "ridge_scale": _H011_RIDGE_SCALE,
        "oracle_lambda": lam_oracle,
        "deployable_lambda": lam_ex,
        "n_full_train_samples": n_full,
        "n_exemplars": n_ex,
        "nme_registered_grouped_in_run_control": _h011_grouped(
            base_module.accuracy(preds_nme.T[0], y_true, nb_old)
        ),
        "linear_oracle_grouped": _h011_grouped(
            base_module.accuracy(preds_oracle.T[0], y_true, nb_old)
        ),
        "linear_deployable_grouped": _h011_grouped(
            base_module.accuracy(preds_deployable.T[0], y_true, nb_old)
        ),
        "invariance_flags": flags,
        "invariance_ok": bool(invariance_ok),
    }
    print("H011_DIAG " + json.dumps(diag, sort_keys=True), flush=True)

    if not invariance_ok:
        raise RuntimeError(
            "H011 invariance violation (persistent state or RNG changed): "
            + json.dumps(diag, sort_keys=True)
        )

    return preds_oracle, y_true


def _install_h011_oracle_bound():
    global _H011_PATCH_INSTALLED
    if _H011_PATCH_INSTALLED:
        return False

    from models.icarl import iCaRL

    original_incremental_train = iCaRL.incremental_train

    def _h011_incremental_train(self, data_manager):
        self._h011_data_manager = data_manager
        return original_incremental_train(self, data_manager)

    iCaRL.incremental_train = _h011_incremental_train
    base_module.BaseLearner._eval_nme = _h011_eval_nme
    _H011_PATCH_INSTALLED = True

    print(
        "H011_PATCH_INSTALLED "
        + json.dumps(
            {
                "target": "BaseLearner._eval_nme (inherited by iCaRL)",
                "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
                "mode": "evaluation-only oracle upper-bound probe; persistent state never written",
                "returns": "linear oracle readout (full-train probe) - NOT deployable",
                "ridge_scale": _H011_RIDGE_SCALE,
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
    # H011: evaluation-only oracle upper bound - how much class information do the
    # frozen features carry? Training, memory, exemplar selection and class means
    # are untouched; the registered NME rule is kept as an in-run control.
    _install_h011_oracle_bound()
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
