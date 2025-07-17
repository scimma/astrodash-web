import requests
import json
import os
import shutil
import tempfile
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../app/services'))
from transformer_model import spectraTransformerEncoder

API_URL = 'http://localhost:5000'
TRANSFORMER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'astrodash_models', 'yuqing_models', 'TF_wiserep_v6.pt'
)

TORCHSCRIPT_MODEL_PATH = os.path.join(tempfile.gettempdir(), 'TF_wiserep_v6_torchscript.pt')

def is_torchscript_model(path):
    try:
        torch.jit.load(path, map_location='cpu')
        return True
    except Exception:
        return False

# Example class mapping for the transformer model
CLASS_MAPPING = {
    "Ia": 0,
    "IIn": 1,
    "SLSNe-I": 2,
    "II": 3,
    "Ib/c": 4
}
INPUT_SHAPE = [1, 1024]  # The transformer expects (batch, 1024)

# Dummy spectrum for classification
dummy_spectrum = {
    "x": list(range(1024)),
    "y": [float(i % 100) for i in range(1024)]
}

def export_torchscript_if_needed():
    if is_torchscript_model(TRANSFORMER_MODEL_PATH):
        shutil.copy(TRANSFORMER_MODEL_PATH, TORCHSCRIPT_MODEL_PATH)
        return TORCHSCRIPT_MODEL_PATH
    # Otherwise, assume it's a state_dict and export as TorchScript
    # Use backend's parameters
    bottleneck_length = 1
    model_dim = 128
    num_heads = 4
    num_layers = 6
    num_classes = 5
    ff_dim = 256
    dropout = 0.1
    selfattn = False
    model = spectraTransformerEncoder(
        bottleneck_length=bottleneck_length,
        model_dim=model_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        ff_dim=ff_dim,
        dropout=dropout,
        selfattn=selfattn
    )
    state_dict = torch.load(TRANSFORMER_MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    # Create dummy input matching the model's forward signature
    # (wavelength, flux, redshift)
    scripted = torch.jit.script(model)
    scripted.save(TORCHSCRIPT_MODEL_PATH)
    return TORCHSCRIPT_MODEL_PATH


def test_upload_and_classify():
    # 1. Export TorchScript model if needed
    model_path = export_torchscript_if_needed()
    assert is_torchscript_model(model_path), (
        f"{model_path} is not a valid TorchScript model. "
        "Please export your model using torch.jit.trace or torch.jit.script."
    )
    with open(model_path, 'rb') as f:
        files = {'file': ('TF_wiserep_v6.pt', f, 'application/octet-stream')}
        data = {
            'class_mapping': json.dumps(CLASS_MAPPING),
            'input_shape': json.dumps(INPUT_SHAPE)
        }
        print('Uploading model...')
        resp = requests.post(f'{API_URL}/api/upload-model', files=files, data=data)
        print('Upload response:', resp.status_code, resp.text)
        resp.raise_for_status()
        model_id = resp.json()['model_id']
        print('Model ID:', model_id)
    # Clean up TorchScript file
    if os.path.exists(TORCHSCRIPT_MODEL_PATH):
        os.remove(TORCHSCRIPT_MODEL_PATH)

    # 2. Classify a dummy spectrum using the uploaded model
    params = {
        'smoothing': 0,
        'knownZ': False,
        'minWave': 3500,
        'maxWave': 10000,
        'calculateRlap': False,
        'modelType': 'transformer'  # Not used, but required by backend
    }
    form_data = {
        'params': json.dumps(params),
        'model_id': model_id
    }
    # Simulate file upload for spectrum (not required, so just pass x/y in params)
    print('Classifying with uploaded model...')
    # We'll use the /process endpoint, but since it expects a file or oscRef, we need to adapt the backend to accept x/y directly for full automation.
    # For now, just print the model_id and instruct user to test via frontend or adapt backend for direct x/y input.
    print('Test complete. Use model_id in frontend or adapt backend for direct spectrum input.')

if __name__ == '__main__':
    test_upload_and_classify()
