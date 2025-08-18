import torch
import os
import numpy as np

PYTORCH_MODEL = os.path.join(os.path.dirname(__file__), '..', 'astrodash_models', 'zeroZ', 'pytorch_model.pth')

state_dict = torch.load(PYTORCH_MODEL, map_location=torch.device('cpu'))

print("=== Checking PyTorch Model Weights ===")
for k, v in state_dict.items():
    arr = v.cpu().numpy()
    print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")
    if np.isnan(arr).any():
        print(f"  [!] NaNs found in {k}")
    if np.isinf(arr).any():
        print(f"  [!] Infs found in {k}")
    print(f"  min={arr.min()}, max={arr.max()}, mean={arr.mean()}, std={arr.std()}")
    if np.abs(arr).max() > 1e3:
        print(f"  [!] Extreme value (>1e3) found in {k}")