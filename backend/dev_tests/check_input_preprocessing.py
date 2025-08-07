import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def check_input_preprocessing():
    """Check if there are differences in input preprocessing"""

    TEST_FILE = "backend/dev_tests/test_spectrum.txt"
    PARAMS = "backend/astrodash_models/zeroZ/training_params.pickle"

    # Load test spectrum
    data = np.loadtxt(TEST_FILE)
    wave, flux = data[:, 0], data[:, 1]

    print(f"Raw spectrum - wave range: {np.min(wave):.2f} to {np.max(wave):.2f}")
    print(f"Raw spectrum - flux range: {np.min(flux):.6f} to {np.max(flux):.6f}")

    # Load parameters
    with open(PARAMS, 'rb') as f:
        import pickle
        pars = pickle.load(f, encoding='latin1')

    w0 = pars['w0']
    w1 = pars['w1']
    nw = pars['nw']

    print(f"\nParameters:")
    print(f"  w0: {w0}")
    print(f"  w1: {w1}")
    print(f"  nw: {nw}")

    # Process using prod_backend
    # Add prod_backend to path
    PROD_BACKEND_PATH = "prod_backend"
    if PROD_BACKEND_PATH not in sys.path:
        sys.path.insert(0, PROD_BACKEND_PATH)

    from app.infrastructure.ml.processors.data_processor import DashSpectrumProcessor
    processor = DashSpectrumProcessor(w0, w1, nw)
    processed_flux_prod, min_idx_prod, max_idx_prod, z_prod = processor.process(wave, flux, z=0.0)

    print(f"\nProd_backend preprocessing:")
    print(f"  Processed flux shape: {processed_flux_prod.shape}")
    print(f"  Min index: {min_idx_prod}, Max index: {max_idx_prod}")
    print(f"  Flux range: {np.min(processed_flux_prod):.6f} to {np.max(processed_flux_prod):.6f}")
    print(f"  Flux mean: {np.mean(processed_flux_prod):.6f}")
    print(f"  Flux std: {np.std(processed_flux_prod):.6f}")

    # Try to process using original astrodash (if possible)
    try:
        # Add astrodash to path
        ASTRODASH_PATH = "astrodash/astrodash"
        if ASTRODASH_PATH not in sys.path:
            sys.path.insert(0, ASTRODASH_PATH)

        from preprocessing import PreProcessSpectrum

        # Create preprocessing object
        preprocessor = PreProcessSpectrum(w0, w1, nw)

        # Process spectrum
        processed_flux_orig = preprocessor.process_spectrum(wave, flux, z=0.0)

        print(f"\nOriginal astrodash preprocessing:")
        print(f"  Processed flux shape: {processed_flux_orig.shape}")
        print(f"  Flux range: {np.min(processed_flux_orig):.6f} to {np.max(processed_flux_orig):.6f}")
        print(f"  Flux mean: {np.mean(processed_flux_orig):.6f}")
        print(f"  Flux std: {np.std(processed_flux_orig):.6f}")

        # Compare the two
        if processed_flux_prod.shape == processed_flux_orig.shape:
            diff = np.abs(processed_flux_prod - processed_flux_orig)
            print(f"\nPreprocessing difference:")
            print(f"  Mean absolute difference: {np.mean(diff):.8f}")
            print(f"  Max absolute difference: {np.max(diff):.8f}")
            print(f"  Correlation: {np.corrcoef(processed_flux_prod.flatten(), processed_flux_orig.flatten())[0,1]:.8f}")

            if np.max(diff) > 1e-6:
                print("  ⚠️  WARNING: Significant preprocessing difference detected!")
            else:
                print("  ✅ Preprocessing matches!")
        else:
            print(f"\nShape mismatch: prod_backend {processed_flux_prod.shape} vs original {processed_flux_orig.shape}")

    except Exception as e:
        print(f"\nCould not run original astrodash preprocessing: {e}")
        print("This might be due to missing dependencies or different preprocessing implementation.")

    # Also check if there are any differences in the preprocessing steps
    print(f"\n=== Detailed Preprocessing Steps ===")

    # Check what the prod_backend processor does
    print("Prod_backend preprocessing steps:")
    print("  1. Normalize spectrum")
    print("  2. Limit wavelength range")
    print("  3. Log wavelength binning")
    print("  4. Continuum removal")
    print("  5. Mean zero")
    print("  6. Apodize")
    print("  7. Zero non-overlap part")

    # Check if these match the original astrodash steps
    print("\nOriginal astrodash preprocessing steps (from code):")
    print("  1. Normalize spectrum")
    print("  2. Limit wavelength range")
    print("  3. Log wavelength binning")
    print("  4. Continuum removal")
    print("  5. Mean zero")
    print("  6. Apodize")
    print("  7. Zero non-overlap part")

    print("\nThe preprocessing steps appear to be the same. The issue might be in:")
    print("  1. Different implementation details")
    print("  2. Different default parameters")
    print("  3. Different numerical precision")
    print("  4. Different order of operations")

if __name__ == "__main__":
    check_input_preprocessing()
