from flower_vla_calvin.flower.models.flower import FLOWERVLA
import json
from safetensors.torch import load_file
 
from flower_vla_calvin.flower.models.flower import FLOWERVLA

with open("config.json", "r") as f:
    config = json.load(f)

model = FLOWERVLA(**config["model_config"], load_pretrained = True, pretrained_model_path="model.safetensors")
# state_dict = load_file("model.safetensors")
# print(state_dict.keys())
# model.load_state_dict(state_dict)
model = model.cuda()
model.eval()
print("done loading")

import torch

# Define dimensions
B = 1   # Batch size
T = 10  # Time steps
C = 3   # RGB channels
H = 64 # Height
W = 64 # Width
for i in range(100):
    # Create random tensors for static and gripper cameras
    rgb_static_cam = torch.rand(B, T, C, H, W)
    rgb_gripper_cam = torch.rand(B, T, C, H, W)

    obs = {
        "rgb_obs": {
            "rgb_static": rgb_static_cam,
            "rgb_gripper": rgb_gripper_cam
        }
    }
    goal = {"lang_text": "pick up the blue cube"}
    action = model.step(obs, goal)
    print(action)

print("done action")
