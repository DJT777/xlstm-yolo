from ultralytics import YOLO
import torch


torch.cuda.is_available()
# Set the default dtype to float64 (double precision)
# torch.set_default_dtype(torch.bfloat16)

import torch

# Assuming the YAML configuration is saved in "custom_yolo.yaml"
model = YOLO("yamls/patch_merge_basic192.yaml")
# # model = model.to(torch.bfloat16)     # Convert to bfloat16
model.model = torch.compile(model.model)