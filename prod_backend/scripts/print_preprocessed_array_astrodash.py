import sys
import os
import numpy as np

# Absolute paths
PROJECT_ROOT = "/home/jesusca/code_personal/astrodash-web"
TEST_FILE = f"{PROJECT_ROOT}/astrodash/astrodash/test_spectrum_file.dat"

# Ensure astrodash is importable
ASTRODASH_PATH = f"{PROJECT_ROOT}/astrodash/astrodash"
if ASTRODASH_PATH not in sys.path:
    sys.path.insert(0, ASTRODASH_PATH)

from preprocessing import PreProcessSpectrum

# Parameters (from training_params or hardcoded for test file)
w0 = 3500.0
w1 = 10000.0
nw = 1024

# Load test spectrum
data = np.loadtxt(TEST_FILE)
wave, flux = data[:, 0], data[:, 1]

# Use the original PreProcessSpectrum
preproc = PreProcessSpectrum(w0, w1, nw)
# Use the same logic as PreProcessing.two_column_data in sn_processing.py
binnedwave, binnedflux, minIndex, maxIndex = preproc.log_wavelength(wave, flux)
newflux, continuum = preproc.continuum_removal(binnedwave, binnedflux, 13, minIndex, maxIndex)
meanzero = preproc.mean_zero(newflux, minIndex, maxIndex)
apodized = preproc.apodize(meanzero, minIndex, maxIndex)
# Median filter with kernel size 1 (no smoothing)
fluxNorm = (apodized - np.min(apodized)) / (np.max(apodized) - np.min(apodized))
fluxNorm = preproc.zero_non_overlap_part(fluxNorm, minIndex, maxIndex, outerVal=0.5)

print("=== Preprocessed Array (original astrodash) ===")
print(f"Shape: {fluxNorm.shape}, dtype: {fluxNorm.dtype}")
print(f"min: {fluxNorm.min()}, max: {fluxNorm.max()}, mean: {fluxNorm.mean()}, std: {fluxNorm.std()}")
print(f"First 10 values: {fluxNorm[:10]}")
print(f"min_idx: {minIndex}, max_idx: {maxIndex}")