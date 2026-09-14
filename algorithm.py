"""
Baseline continual learning algorithm: iCaRL on CIFAR-100.

This file contains the epoch and hyperparameter configuration for iCaRL.
The actual iCaRL implementation is in the PyCIL repository (models/icarl.py).

Agents should modify the hyperparameters below, or replace the model_name
with a different continual learning algorithm from PyCIL (e.g., 'der', 'foster',
'memo', 'ewc', 'lwf', etc.).

To make deeper changes, agents can also modify models/icarl.py directly,
but must keep the BaseLearner interface (incremental_train, eval_task, after_task).
"""


_H009_PATCH_INSTALLED = False


def _install_new_class_full_data_prototype_patch():
    """H009 minimal intervention (algorithm.py-only).

    Keep training, exemplar selection and the stored exemplars unchanged, but
    after the normal iCaRL exemplar construction recompute the NME prototypes
    of the classes introduced at this stage from *all* available training
    samples of those classes, reusing the existing normalized-vector mean
    procedure (L2-normalize each feature -> mean -> L2-normalize).

    Old-class means are still produced from the stored exemplars by
    `_reduce_exemplar`, and `_data_memory`/`_targets_memory` are untouched, so
    the 2000-exemplar budget and the exemplar selection logic are preserved.

    The extra feature extraction is wrapped in a global-RNG snapshot/restore.
    Building or iterating any DataLoader draws `_base_seed` from the *global*
    torch generator (`torch.utils.data.dataloader._BaseDataLoaderIter.__init__`),
    so without this guard the added loaders would silently shift the shuffle
    order of every later training loader and change the CNN trajectory, which
    is the registered invariance control.
    """
    global _H009_PATCH_INSTALLED
    if _H009_PATCH_INSTALLED:
        return

    import logging

    import numpy as np
    import torch
    import models.base as base_module

    original_construct_exemplar = base_module.BaseLearner._construct_exemplar

    def _construct_exemplar_full_data_means(self, data_manager, m):
        original_construct_exemplar(self, data_manager, m)

        cpu_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            for class_idx in range(self._known_classes, self._total_classes):
                _, _, class_dset = data_manager.get_dataset(
                    np.arange(class_idx, class_idx + 1),
                    source="train",
                    mode="test",
                    ret_data=True,
                )
                class_loader = base_module.DataLoader(
                    class_dset,
                    batch_size=base_module.batch_size,
                    shuffle=False,
                    num_workers=4,
                )
                vectors, _ = self._extract_vectors(class_loader)
                vectors = (
                    vectors.T
                    / (np.linalg.norm(vectors.T, axis=0) + base_module.EPSILON)
                ).T
                mean = np.mean(vectors, axis=0)
                mean = mean / np.linalg.norm(mean)
                self._class_means[class_idx, :] = mean
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        logging.info(
            "[H009] full-data prototypes for classes {}-{} ({} classes); "
            "rng_state_restored={}".format(
                self._known_classes,
                self._total_classes,
                self._total_classes - self._known_classes,
                bool(torch.get_rng_state().equal(cpu_rng)),
            )
        )

    base_module.BaseLearner._construct_exemplar = _construct_exemplar_full_data_means
    _H009_PATCH_INSTALLED = True


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
    # H009: recompute only the newly introduced classes' NME prototypes from
    # all available training samples; training, memory budget and exemplar
    # selection are unchanged.
    _install_new_class_full_data_prototype_patch()
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
