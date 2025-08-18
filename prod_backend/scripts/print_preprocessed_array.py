import sys
import os
import numpy as np
import pickle

# Absolute paths
PROJECT_ROOT = "/home/jesusca/code_personal/astrodash-web"
TEST_FILE = f"{PROJECT_ROOT}/astrodash/astrodash/test_spectrum_file.dat"
PARAMS = f"{PROJECT_ROOT}/backend/astrodash_models/zeroZ/training_params.pickle"
PROD_BACKEND_PATH = f"{PROJECT_ROOT}/prod_backend"
if PROD_BACKEND_PATH not in sys.path:
    sys.path.insert(0, PROD_BACKEND_PATH)

from app.infrastructure.ml.processors.data_processor import DashSpectrumProcessor

# Load parameters
with open(PARAMS, 'rb') as f:
    pars = pickle.load(f, encoding='latin1')
w0 = pars['w0']
w1 = pars['w1']
nw = pars['nw']

# Load test spectrum
data = np.loadtxt(TEST_FILE)
wave, flux = data[:, 0], data[:, 1]

# Preprocess
processor = DashSpectrumProcessor(w0, w1, nw)
processed_flux, min_idx, max_idx, z = processor.process(wave, flux, z=0.0)

print("=== Preprocessed Array (prod_backend) ===")
print(f"Shape: {processed_flux.shape}, dtype: {processed_flux.dtype}")
print(f"min: {processed_flux.min()}, max: {processed_flux.max()}, mean: {processed_flux.mean()}, std: {processed_flux.std()}")
print(f"First 10 values: {processed_flux[:10]}")
print(f"min_idx: {min_idx}, max_idx: {max_idx}, z: {z}")