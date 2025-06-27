import numpy as np
import sys
import os

# Add the backend directory to sys.path to use local astrodash code
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Old astrodash-based method ---
from astrodash.astrodash.data.read_binned_templates import load_templates, get_templates, combined_sn_and_host_data

# --- New backend method ---
from app.services.astrodash_backend import get_training_parameters, load_template_spectrum

# Path to the .npz file (update if needed)
npz_path = os.path.join(os.path.dirname(__file__), '..', 'astrodash_models', 'sn_and_host_templates.npz')

# Load the .npz file
data = np.load(npz_path, allow_pickle=True)
snTemplates = data['snTemplates'].item() if 'snTemplates' in data else data['arr_0'].item()

def print_structure():
    print("\n===== snTemplates Structure =====\n")
    for sn_type in snTemplates:
        print(f"SN Type: {sn_type}")
        age_bins = snTemplates[sn_type]
        for age_bin in age_bins:
            print(f"  Age Bin: {age_bin}")
            entry = age_bins[age_bin]
            if not isinstance(entry, dict):
                print(f"    [ERROR] Entry is not a dict: {type(entry)}")
                continue
            for key in entry:
                value = entry[key]
                try:
                    if isinstance(value, np.ndarray):
                        print(f"    Key: {key}, Type: np.ndarray, Shape: {value.shape}, Dtype: {value.dtype}")
                    elif isinstance(value, list):
                        print(f"    Key: {key}, Type: list, Length: {len(value)}")
                    else:
                        print(f"    Key: {key}, Type: {type(value)}")
                except Exception as e:
                    print(f"    Key: {key}, [ERROR accessing value]: {e}")
            # Try to access snInfo and print its shape/type
            try:
                snInfo = entry.get('snInfo', None)
                if snInfo is not None:
                    print(f"    snInfo: type={type(snInfo)}, shape={getattr(snInfo, 'shape', 'N/A')}")
            except Exception as e:
                print(f"    [ERROR accessing snInfo]: {e}")
            # Try to access wave/flux if present
            for arr_key in ['wave', 'flux']:
                try:
                    arr = entry.get(arr_key, None)
                    if arr is not None:
                        print(f"    {arr_key}: type={type(arr)}, shape={getattr(arr, 'shape', 'N/A')}")
                except Exception as e:
                    print(f"    [ERROR accessing {arr_key}]: {e}")
            print()

if __name__ == "__main__":
    print_structure()
