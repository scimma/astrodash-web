import sys
import os
import numpy as np
import torch
import pickle

# Absolute paths
PROJECT_ROOT = "/home/jesusca/code_personal/astrodash-web"
TEST_FILE = f"{PROJECT_ROOT}/astrodash/astrodash/test_spectrum_file.dat"
PARAMS = f"{PROJECT_ROOT}/backend/astrodash_models/zeroZ/training_params.pickle"
PYTORCH_MODEL = f"{PROJECT_ROOT}/backend/astrodash_models/zeroZ/pytorch_model.pth"
TF_MODEL = f"{PROJECT_ROOT}/backend/astrodash_models/zeroZ/tensorflow_model.ckpt"

# Ensure astrodash is importable
ASTRODASH_PATH = f"{PROJECT_ROOT}/astrodash/astrodash"
if ASTRODASH_PATH not in sys.path:
    sys.path.insert(0, ASTRODASH_PATH)

# Ensure prod_backend is importable
PROD_BACKEND_PATH = f"{PROJECT_ROOT}/prod_backend"
if PROD_BACKEND_PATH not in sys.path:
    sys.path.insert(0, PROD_BACKEND_PATH)

def load_and_preprocess_spectrum():
    """Load and preprocess the test spectrum using prod_backend pipeline"""
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

    # Preprocess using prod_backend
    processor = DashSpectrumProcessor(w0, w1, nw)
    processed_flux, min_idx, max_idx, z = processor.process(wave, flux, z=0.0)

    return processed_flux.astype(np.float32), min_idx, max_idx

def run_pytorch_forward_pass(processed_flux):
    """Run forward pass through PyTorch model"""
    # Load PyTorch model
    state_dict = torch.load(PYTORCH_MODEL, map_location=torch.device('cpu'))

    # Determine n_types from the output layer
    if 'output.weight' in state_dict:
        n_types = state_dict['output.weight'].shape[0]  # New architecture
    else:
        n_types = state_dict['classifier.3.weight'].shape[0]  # Old architecture

    # Import the corrected architecture from the conversion script
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from convert_model import AstroDashPyTorchNet
    model = AstroDashPyTorchNet(n_types)
    model.load_state_dict(state_dict)
    model.eval()

    # Prepare input
    input_tensor = torch.from_numpy(processed_flux).float().reshape(1, -1)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_tensor)
        softmax = outputs[0].cpu().numpy()

    return softmax

def run_tensorflow_forward_pass(processed_flux):
    """Run forward pass through TensorFlow model"""
    try:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()

        # Load TensorFlow model
        saver = tf.train.import_meta_graph(TF_MODEL + '.meta')

        with tf.Session() as sess:
            saver.restore(sess, TF_MODEL)

            # Get the input and output tensors
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("Placeholder:0")  # Input placeholder
            keep_prob = graph.get_tensor_by_name("Placeholder_2:0")  # Dropout keep probability
            y_conv = graph.get_tensor_by_name("Softmax:0")  # Output softmax

            # Prepare input
            input_data = processed_flux.reshape(1, -1)

            # Forward pass (keep_prob=1.0 for inference, no dropout)
            tf_output = sess.run(y_conv, feed_dict={x: input_data, keep_prob: 1.0})
            return tf_output[0]

    except Exception as e:
        print(f"TensorFlow forward pass failed: {e}")
        return None

def compare_outputs(pytorch_output, tf_output):
    """Compare outputs from both models"""
    print("=== Model Output Comparison ===")

    if tf_output is None:
        print("TensorFlow output not available, only showing PyTorch output")
        pytorch_top10 = np.argsort(pytorch_output)[::-1][:10]
        print("PyTorch top 10 outputs:")
        for i, idx in enumerate(pytorch_top10):
            print(f"  {i+1}: index {idx}, probability {pytorch_output[idx]:.6f}")
        return

    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"TensorFlow output shape: {tf_output.shape}")

    # Compare shapes
    if pytorch_output.shape != tf_output.shape:
        print(f"WARNING: Output shapes don't match! PyTorch: {pytorch_output.shape}, TF: {tf_output.shape}")
        min_len = min(len(pytorch_output), len(tf_output))
        pytorch_output = pytorch_output[:min_len]
        tf_output = tf_output[:min_len]

    # Get top 10 from both
    pytorch_top10 = np.argsort(pytorch_output)[::-1][:10]
    tf_top10 = np.argsort(tf_output)[::-1][:10]

    print("\nPyTorch top 10 outputs:")
    for i, idx in enumerate(pytorch_top10):
        print(f"  {i+1}: index {idx}, probability {pytorch_output[idx]:.6f}")

    print("\nTensorFlow top 10 outputs:")
    for i, idx in enumerate(tf_top10):
        print(f"  {i+1}: index {idx}, probability {tf_output[idx]:.6f}")

    # Compare differences
    print(f"\nOutput differences:")
    print(f"  Mean absolute difference: {np.mean(np.abs(pytorch_output - tf_output)):.6f}")
    print(f"  Max absolute difference: {np.max(np.abs(pytorch_output - tf_output)):.6f}")
    print(f"  Correlation: {np.corrcoef(pytorch_output, tf_output)[0,1]:.6f}")

    # Check if top predictions match
    pytorch_top5 = set(pytorch_top10[:5])
    tf_top5 = set(tf_top10[:5])
    overlap = len(pytorch_top5.intersection(tf_top5))
    print(f"  Top 5 overlap: {overlap}/5 predictions match")

if __name__ == "__main__":
    print("Loading and preprocessing spectrum...")
    processed_flux, min_idx, max_idx = load_and_preprocess_spectrum()
    print(f"Preprocessed spectrum shape: {processed_flux.shape}")

    print("\nRunning PyTorch forward pass...")
    pytorch_output = run_pytorch_forward_pass(processed_flux)

    print("\nRunning TensorFlow forward pass...")
    tf_output = run_tensorflow_forward_pass(processed_flux)

    print("\nComparing outputs...")
    compare_outputs(pytorch_output, tf_output)
