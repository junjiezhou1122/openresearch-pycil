"""
H027 CPU preflight check for Feature Cosine Distillation.
"""
from __future__ import annotations

import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from algorithm import get_pycil_config

cfg = get_pycil_config()
print("H027_CONFIG:", cfg)

assert cfg["memory_size"] == 6000, f"Expected memory_size=6000, got {cfg['memory_size']}"
assert cfg["init_cls"] == 50, f"Expected init_cls=50, got {cfg['init_cls']}"
assert cfg["increment"] == 10, f"Expected increment=10, got {cfg['increment']}"
assert cfg["init_epoch"] == 100, f"Expected init_epoch=100, got {cfg['init_epoch']}"
assert cfg["epochs"] == 50, f"Expected epochs=50, got {cfg['epochs']}"
assert cfg["convnet_type"] == "resnet18", f"Expected convnet_type=resnet18, got {cfg['convnet_type']}"
assert cfg["seed"] == [1993], f"Expected seed=[1993], got {cfg['seed']}"
assert cfg["dataset"] == "cifar100", f"Expected dataset=cifar100, got {cfg['dataset']}"

print("PREFLIGHT_CHECK config_asserts ok=True")

import models.icarl as icarl_module
import torch
import torch.nn.functional as F

assert icarl_module.iCaRL._update_representation.__name__ == "_h027_update_representation", (
    f"Expected patched _h027_update_representation, got {icarl_module.iCaRL._update_representation.__name__}"
)
print("PREFLIGHT_CHECK update_representation_patched ok=True")

# Test tensor math
cur_feat = torch.randn(4, 512)
old_feat = torch.randn(4, 512)
loss_feat = (1.0 - F.cosine_similarity(cur_feat, old_feat, dim=-1)).mean()
assert not torch.isnan(loss_feat), "Cosine feature loss returned NaN"
assert loss_feat.item() >= 0.0, "Cosine feature loss is negative"

# Identity test: identical features must have loss 0
loss_zero = (1.0 - F.cosine_similarity(cur_feat, cur_feat, dim=-1)).mean()
assert abs(loss_zero.item()) < 1e-6, f"Identical features loss != 0: {loss_zero.item()}"

print("PREFLIGHT_CHECK tensor_loss_math ok=True")
print("PREFLIGHT_OK 8/8")
