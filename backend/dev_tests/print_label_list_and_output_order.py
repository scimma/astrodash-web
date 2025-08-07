import os
import pickle
import numpy as np
import torch

# Paths (adjust if needed)
ORIG_PARAMS = os.path.join(os.path.dirname(__file__), '..', 'astrodash_models', 'zeroZ', 'training_params.pickle')
PYTORCH_MODEL = os.path.join(os.path.dirname(__file__), '..', 'astrodash_models', 'zeroZ', 'pytorch_model.pth')

# --- Print label list from original astrodash logic ---
def print_original_label_list():
    print("\n=== Original AstroDash Label List ===")
    with open(ORIG_PARAMS, 'rb') as f:
        pars = pickle.load(f, encoding='latin1')
    typeList = pars['typeList']
    minAge = pars['minAge']
    maxAge = pars['maxAge']
    ageBinSize = pars['ageBinSize']
    # Age binning logic (copied from astrodash)
    def age_labels():
        ageLabels = []
        ageBinPrev = 0
        ageLabelMin = minAge
        for age in np.arange(minAge, maxAge, 0.5):
            ageBin = int(round(age / ageBinSize)) - int(round(minAge / ageBinSize))
            if ageBin != ageBinPrev:
                ageLabelMax = int(round(age))
                ageLabels.append(f"{int(ageLabelMin)} to {ageLabelMax}")
                ageLabelMin = ageLabelMax
            ageBinPrev = ageBin
        ageLabels.append(f"{int(ageLabelMin)} to {int(maxAge)}")
        return ageLabels
    labels = []
    for tType in typeList:
        for ageLabel in age_labels():
            labels.append(f"{tType}: {ageLabel}")
    print(f"Total labels: {len(labels)}")
    for i, label in enumerate(labels[:10]):
        print(f"{i}: {label}")
    if len(labels) > 10:
        print("...")
    return labels

# --- Print label list from prod_backend logic ---
def print_backend_label_list():
    print("\n=== prod_backend Label List ===")
    # Simulate prod_backend label logic
    with open(ORIG_PARAMS, 'rb') as f:
        pars = pickle.load(f, encoding='latin1')
    typeList = pars['typeList']
    minAge = pars['minAge']
    maxAge = pars['maxAge']
    ageBinSize = pars['ageBinSize']
    def age_labels():
        ageLabels = []
        ageBinPrev = 0
        ageLabelMin = minAge
        for age in np.arange(minAge, maxAge, 0.5):
            ageBin = int(round(age / ageBinSize)) - int(round(minAge / ageBinSize))
            if ageBin != ageBinPrev:
                ageLabelMax = int(round(age))
                ageLabels.append(f"{int(ageLabelMin)} to {ageLabelMax}")
                ageLabelMin = ageLabelMax
            ageBinPrev = ageBin
        ageLabels.append(f"{int(ageLabelMin)} to {int(maxAge)}")
        return ageLabels
    labels = []
    for tType in typeList:
        for ageLabel in age_labels():
            labels.append(f"{tType}: {ageLabel}")
    print(f"Total labels: {len(labels)}")
    for i, label in enumerate(labels[:10]):
        print(f"{i}: {label}")
    if len(labels) > 10:
        print("...")
    return labels

# --- Print output layer shape and check order ---
def print_pytorch_output_layer():
    print("\n=== PyTorch Model Output Layer ===")
    state_dict = torch.load(PYTORCH_MODEL, map_location=torch.device('cpu'))
    out_weight = state_dict['classifier.3.weight']
    out_bias = state_dict['classifier.3.bias']
    print(f"Output layer weight shape: {out_weight.shape}")
    print(f"Output layer bias shape: {out_bias.shape}")
    print(f"First 5 bias values: {out_bias[:5].numpy()}")
    print(f"First 5 weight vector norms: {[float(out_weight[i].norm()) for i in range(5)]}")
    return out_weight, out_bias

if __name__ == "__main__":
    orig_labels = print_original_label_list()
    backend_labels = print_backend_label_list()
    if orig_labels == backend_labels:
        print("\nLabel lists are IDENTICAL.")
    else:
        print("\nLabel lists are DIFFERENT! Check for order or content mismatches.")
    print_pytorch_output_layer()