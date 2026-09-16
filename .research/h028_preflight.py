"""
H028 CPU preflight check for Lower-Layer Freezing.
"""
from __future__ import annotations

import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from algorithm import get_pycil_config

cfg = get_pycil_config()
print("H028_CONFIG:", cfg)

assert cfg["memory_size"] == 6000, f"Expected memory_size=6000, got {cfg['memory_size']}"
assert cfg["init_cls"] == 50, f"Expected init_cls=50, got {cfg['init_cls']}"
assert cfg["increment"] == 10, f"Expected increment=10, got {cfg['increment']}"
assert cfg["convnet_type"] == "resnet18", f"Expected convnet_type=resnet18, got {cfg['convnet_type']}"
assert cfg["seed"] == [1993], f"Expected seed=[1993], got {cfg['seed']}"
print("PREFLIGHT_CHECK config_asserts ok=True")

import models.icarl as icarl_module
import utils.factory as factory
import utils.data_manager as data_manager

assert icarl_module.iCaRL.after_task.__name__ == "_h028_after_task", (
    f"Expected patched _h028_after_task, got {icarl_module.iCaRL.after_task.__name__}"
)
print("PREFLIGHT_CHECK after_task_patched ok=True")

model = factory.get_model("icarl", cfg)
model._cur_task = 0

# Before task 0 finishes: conv1 and layer1 should require grad
for p in model._network.convnet.conv1.parameters():
    assert p.requires_grad is True, "conv1 should require grad before after_task"
for p in model._network.convnet.layer1.parameters():
    assert p.requires_grad is True, "layer1 should require grad before after_task"

# Call after_task
model.after_task()

# After task 0: conv1 and layer1 must be frozen
for p in model._network.convnet.conv1.parameters():
    assert p.requires_grad is False, "conv1 should NOT require grad after after_task"
for p in model._network.convnet.layer1.parameters():
    assert p.requires_grad is False, "layer1 should NOT require grad after after_task"

# Higher layers must STILL require grad
for p in model._network.convnet.layer2.parameters():
    assert p.requires_grad is True, "layer2 must still require grad"
for p in model._network.convnet.layer3.parameters():
    assert p.requires_grad is True, "layer3 must still require grad"
for p in model._network.convnet.layer4.parameters():
    assert p.requires_grad is True, "layer4 must still require grad"

# Check that net.train() does not revert conv1/layer1 to training=True
model._network.train()
assert model._network.convnet.conv1.training is False, "conv1 should stay in eval mode after net.train()"
assert model._network.convnet.layer1.training is False, "layer1 should stay in eval mode after net.train()"
assert model._network.convnet.layer2.training is True, "layer2 should be in train mode after net.train()"

print("PREFLIGHT_CHECK freezing_behavior ok=True")
print("PREFLIGHT_OK 8/8")
