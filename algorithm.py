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
H012: is any of the oracle headroom REACHABLE inside the learner's information set?
-------------------------------------------------------------------------------
Everything about training is identical to the verified H001 configuration
(iCaRL + ResNet-18, init_epoch=100, epochs=50, seed 1993, memory_size=2000).

Where this comes from. Wave 2 measured two bounds on the accepted metric:
  * H010 (valid/refuted): a closed-form ridge readout fitted on the SAME stored
    exemplars cannot beat the registered nearest-mean rule (-0.66 pp aggregate).
  * H011 (valid/supported): a closed-form ridge oracle fitted on the FULL training
    split of all seen classes beats the registered NME at every incremental stage
    (+0.94/+2.66/+3.13/+3.67/+3.77 pp; +2.36 pp aggregate), and the gain is
    entirely OLD-class retention while NEW-class accuracy regresses.
So roughly +2.36 pp of headroom exists in the frozen features, but H011's oracle
used data a class-incremental learner does not have. H012 asks the decision
question: is any of that headroom reachable with the information a learner
ACTUALLY holds at evaluation time?

The deployable information set at stage t is exactly:
  * the stored exemplar memory for every class seen so far (the same 2000-exemplar
    budget the registered NME summarises into class means), plus
  * the FULL training split of the classes introduced at this stage (the current
    task's data, which is available during the stage).

H012 fits a class-balanced closed-form ridge readout on exactly that set and
scores it against the registered nearest-mean rule in the same pass.

  * If it beats the registered NME coherently, the accepted metric is improvable
    without any extra memory -- a deployable readout improvement.
  * If it does not, the H011 headroom is unreachable inside the class-incremental
    constraint, and the registered nearest-mean rule is already at the
    information-set frontier.

Scope / auditability:
  * Closed form (no optimiser, no gradient step, no iteration). Class balancing is
    applied as per-sample weights 1/count(class), so the old-class majority of the
    H011 oracle cannot drive the fit.
  * Regularisation is fixed a priori at ``lam = 1e-3 * trace(X^T X) / d`` on the
    UNWEIGHTED Gram matrix, never tuned on the test split.
  * Persistent state, training, memories, fc weights and the CPU/CUDA RNG stream
    are never written; before/after sha256 equality is printed per stage and any
    mismatch raises.
  * The registered nearest-mean rule is scored in the SAME pass as the in-run
    control, so the patch is provably a readout swap only.
  * Returned prediction = the deployable class-balanced readout.
"""

import hashlib
import json

import numpy as np
import torch

import models.base as base_module

_H012_PATCH_INSTALLED = False
_H012_EPS = base_module.EPSILON
_H012_RIDGE_SCALE = 1e-3


def _h012_sha_array(value):
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _h012_sha_tensor(tensor):
    return hashlib.sha256(
        tensor.detach().to("cpu").contiguous().numpy().tobytes()
    ).hexdigest()


def _h012_sha_network(net):
    module = net.module if isinstance(net, torch.nn.DataParallel) else net
    digest = hashlib.sha256()
    state = module.state_dict()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(_h012_sha_tensor(state[name]).encode("ascii"))
    return digest.hexdigest()


def _h012_grouped(grouped):
    return {str(k): (float(v) if v is not None else None) for k, v in grouped.items()}


def _h012_normalize(x):
    return (x.T / (np.linalg.norm(x.T, axis=0) + _H012_EPS)).T


def _h012_balanced_ridge(X, y, n_classes):
    X = _h012_normalize(X)
    Y = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    Y[np.arange(X.shape[0]), y.astype(int)] = 1.0
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(np.float64)
    sample_w = np.array(
        [1.0 / counts[int(c)] if counts[int(c)] > 0 else 0.0 for c in y], dtype=np.float64
    )
    gram = X.T @ (X * sample_w[:, None])
    lam = float(_H012_RIDGE_SCALE * np.trace(gram) / gram.shape[0])
    rhs = X.T @ (Y * sample_w[:, None])
    W = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), rhs)
    return W, lam


def _h012_eval_nme(self, loader, class_means):
    """H012: registered NME vs a class-balanced readout on the deployable information set."""
    import torch.nn as nn

    from scipy.spatial.distance import cdist

    data_manager = getattr(self, "_h012_data_manager", None)
    nb_old = int(self._known_classes)
    nb_total = int(self._total_classes)
    topk = int(self.topk)

    net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network

    persistent = class_means
    cm_before = _h012_sha_array(persistent)
    data_before = _h012_sha_array(self._data_memory)
    targets_before = _h012_sha_array(self._targets_memory)
    weight_before = _h012_sha_tensor(net.fc.weight)
    bias_before = None if net.fc.bias is None else _h012_sha_tensor(net.fc.bias)
    network_before = _h012_sha_network(self._network)

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    feat_blocks, label_blocks = [], []
    n_mem = n_new_full = 0
    try:
        if data_manager is not None:
            # (1) stored exemplar memory (the learner's only old-class data)
            if len(self._data_memory) != 0:
                mem_dset = data_manager.get_dataset(
                    [],
                    source="train",
                    mode="test",
                    appendent=(self._data_memory, self._targets_memory),
                )
                mem_loader = base_module.DataLoader(
                    mem_dset, batch_size=base_module.batch_size, shuffle=False, num_workers=4
                )
                mx, my = self._extract_vectors(mem_loader)
                feat_blocks.append(mx)
                label_blocks.append(my)
                n_mem = int(len(my))
            # (2) the current stage's FULL training split (new classes only)
            if nb_total > nb_old:
                new_dset = data_manager.get_dataset(
                    np.arange(nb_old, nb_total), source="train", mode="test"
                )
                new_loader = base_module.DataLoader(
                    new_dset, batch_size=base_module.batch_size, shuffle=False, num_workers=4
                )
                nx, ny = self._extract_vectors(new_loader)
                feat_blocks.append(nx)
                label_blocks.append(ny)
                n_new_full = int(len(ny))
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
    vectors_t = _h012_normalize(vectors_t)

    preds_nme = np.argsort(cdist(persistent, vectors_t, "sqeuclidean").T, axis=1)[:, :topk]

    deploy_ok = False
    lam = None
    preds_deploy = preds_nme
    if feat_blocks:
        X = np.concatenate(feat_blocks)
        y = np.concatenate(label_blocks)
        W, lam = _h012_balanced_ridge(X, y, nb_total)
        preds_deploy = np.argsort(-(vectors_t @ W), axis=1)[:, :topk]
        deploy_ok = True

    cm_after = _h012_sha_array(persistent)
    data_after = _h012_sha_array(self._data_memory)
    targets_after = _h012_sha_array(self._targets_memory)
    weight_after = _h012_sha_tensor(net.fc.weight)
    bias_after = None if net.fc.bias is None else _h012_sha_tensor(net.fc.bias)
    network_after = _h012_sha_network(self._network)

    flags = {
        "class_means_sha_unchanged": cm_after == cm_before,
        "data_memory_sha_unchanged": data_after == data_before,
        "targets_memory_sha_unchanged": targets_after == targets_before,
        "fc_weight_sha_unchanged": weight_after == weight_before,
        "fc_bias_sha_unchanged": bias_after == bias_before,
        "network_sha_unchanged": network_after == network_before,
        "rng_cpu_restored": rng_cpu_restored,
        "rng_cuda_restored": rng_cuda_restored,
        "deployable_readout_fitted": deploy_ok,
    }
    invariance_ok = all(v for v in flags.values() if v is not None)

    diag = {
        "stage": int(self._cur_task),
        "nb_old": nb_old,
        "nb_total": nb_total,
        "ridge_scale": _H012_RIDGE_SCALE,
        "deployable_lambda": lam,
        "n_memory_samples": n_mem,
        "n_new_class_full_samples": n_new_full,
        "nme_registered_grouped_in_run_control": _h012_grouped(
            base_module.accuracy(preds_nme.T[0], y_true, nb_old)
        ),
        "deployable_balanced_grouped": _h012_grouped(
            base_module.accuracy(preds_deploy.T[0], y_true, nb_old)
        ),
        "invariance_flags": flags,
        "invariance_ok": bool(invariance_ok),
    }
    print("H012_DIAG " + json.dumps(diag, sort_keys=True), flush=True)

    if not invariance_ok:
        raise RuntimeError(
            "H012 invariance violation (persistent state or RNG changed): "
            + json.dumps(diag, sort_keys=True)
        )

    return preds_deploy, y_true


def _install_h012_deployable_bound():
    global _H012_PATCH_INSTALLED
    if _H012_PATCH_INSTALLED:
        return False

    from models.icarl import iCaRL

    original_incremental_train = iCaRL.incremental_train

    def _h012_incremental_train(self, data_manager):
        self._h012_data_manager = data_manager
        return original_incremental_train(self, data_manager)

    iCaRL.incremental_train = _h012_incremental_train
    base_module.BaseLearner._eval_nme = _h012_eval_nme
    _H012_PATCH_INSTALLED = True

    print(
        "H012_PATCH_INSTALLED "
        + json.dumps(
            {
                "target": "BaseLearner._eval_nme (inherited by iCaRL)",
                "base_sha": "839aece6389ba6948141c95cc969fcb1d11b681a",
                "mode": "evaluation-only deployable bounded readout; persistent state never written",
                "information_set": "stored exemplar memory + current-stage full train split",
                "readout": "class-balanced closed-form ridge",
                "ridge_scale": _H012_RIDGE_SCALE,
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
    # H012: evaluation-only class-balanced readout on the learner's actual
    # information set (exemplar memory + current-stage full data). Training,
    # memory, exemplar selection and class means are untouched; the registered
    # NME rule is kept as an in-run control.
    _install_h012_deployable_bound()
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
