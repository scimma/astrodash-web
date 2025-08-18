import sys
import os
import numpy as np
import torch
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_and_preprocess_spectrum():
    """Load and preprocess the test spectrum using prod_backend pipeline"""
    # Add prod_backend to path
    PROD_BACKEND_PATH = "prod_backend"
    if PROD_BACKEND_PATH not in sys.path:
        sys.path.insert(0, PROD_BACKEND_PATH)

    from app.infrastructure.ml.processors.data_processor import DashSpectrumProcessor

    # Load parameters
    PARAMS = "backend/astrodash_models/zeroZ/training_params.pickle"
    TEST_FILE = "backend/dev_tests/test_spectrum.txt"

    with open(PARAMS, 'rb') as f:
        import pickle
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

def compare_intermediate_activations():
    """Compare intermediate activations between TF and PyTorch models"""

    # Load preprocessed spectrum
    processed_flux, min_idx, max_idx = load_and_preprocess_spectrum()
    print(f"Preprocessed spectrum shape: {processed_flux.shape}")
    print(f"Spectrum stats - min: {np.min(processed_flux):.6f}, max: {np.max(processed_flux):.6f}, mean: {np.mean(processed_flux):.6f}")

    # Load PyTorch model
    PYTORCH_MODEL = "backend/astrodash_models/zeroZ/pytorch_model.pth"
    state_dict = torch.load(PYTORCH_MODEL, map_location=torch.device('cpu'))
    n_types = state_dict['output.weight'].shape[0]

    from backend.dev_tests.convert_model import AstroDashPyTorchNet
    pytorch_model = AstroDashPyTorchNet(n_types)
    pytorch_model.load_state_dict(state_dict)
    pytorch_model.eval()

    # Prepare input
    input_tensor = torch.from_numpy(processed_flux).float().reshape(1, -1)

    print("\n=== PyTorch Intermediate Activations ===")

    # Hook to capture intermediate activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook

    # Register hooks
    pytorch_model.layer1.register_forward_hook(get_activation('layer1_output'))
    pytorch_model.layer2.register_forward_hook(get_activation('layer2_output'))
    pytorch_model.fc1.register_forward_hook(get_activation('fc1_output'))

    # Forward pass
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor)
        pytorch_final = pytorch_output[0].cpu().numpy()

    print("PyTorch activations:")
    for name, activation in activations.items():
        print(f"  {name}: shape={activation.shape}, min={np.min(activation):.6f}, max={np.max(activation):.6f}, mean={np.mean(activation):.6f}")

    print(f"  Final output: shape={pytorch_final.shape}, min={np.min(pytorch_final):.6f}, max={np.max(pytorch_final):.6f}, mean={np.mean(pytorch_final):.6f}")

    # Now try to get TensorFlow intermediate activations
    print("\n=== TensorFlow Intermediate Activations ===")

    TF_MODEL = "backend/astrodash_models/zeroZ/tensorflow_model.ckpt"

    try:
        # Load TensorFlow model
        saver = tf.train.import_meta_graph(TF_MODEL + '.meta')

        with tf.Session() as sess:
            saver.restore(sess, TF_MODEL)
            graph = tf.get_default_graph()

            # Get tensors
            x = graph.get_tensor_by_name("Placeholder:0")
            keep_prob = graph.get_tensor_by_name("Placeholder_2:0")
            y_conv = graph.get_tensor_by_name("Softmax:0")

            # Try to get intermediate tensors
            try:
                # Get intermediate tensors by name - capture AFTER max pooling to match PyTorch
                layer1_output = graph.get_tensor_by_name("MaxPool:0")  # After first conv + relu + maxpool
                layer2_output = graph.get_tensor_by_name("MaxPool_1:0")  # After second conv + relu + maxpool
                fc1_output = graph.get_tensor_by_name("Relu_2:0")  # After first FC + relu

                # Run forward pass
                input_data = processed_flux.reshape(1, -1)
                tf_intermediates = sess.run([layer1_output, layer2_output, fc1_output, y_conv],
                                          feed_dict={x: input_data, keep_prob: 1.0})

                print("TensorFlow activations:")
                print(f"  layer1_output: shape={tf_intermediates[0].shape}, min={np.min(tf_intermediates[0]):.6f}, max={np.max(tf_intermediates[0]):.6f}, mean={np.mean(tf_intermediates[0]):.6f}")
                print(f"  layer2_output: shape={tf_intermediates[1].shape}, min={np.min(tf_intermediates[1]):.6f}, max={np.max(tf_intermediates[1]):.6f}, mean={np.mean(tf_intermediates[1]):.6f}")
                print(f"  fc1_output: shape={tf_intermediates[2].shape}, min={np.min(tf_intermediates[2]):.6f}, max={np.max(tf_intermediates[2]):.6f}, mean={np.mean(tf_intermediates[2]):.6f}")
                print(f"  Final output: shape={tf_intermediates[3].shape}, min={np.min(tf_intermediates[3]):.6f}, max={np.max(tf_intermediates[3]):.6f}, mean={np.mean(tf_intermediates[3]):.6f}")

                # Compare activations
                print("\n=== Activation Comparison ===")

                # Compare layer1 outputs
                pytorch_layer1 = activations['layer1_output']
                tf_layer1 = tf_intermediates[0]

                if pytorch_layer1.shape == tf_layer1.shape:
                    diff = np.abs(pytorch_layer1 - tf_layer1)
                    print(f"Layer1 difference - mean: {np.mean(diff):.8f}, max: {np.max(diff):.8f}")
                    print(f"Layer1 correlation: {np.corrcoef(pytorch_layer1.flatten(), tf_layer1.flatten())[0,1]:.8f}")
                else:
                    print(f"Layer1 shape mismatch: PyTorch {pytorch_layer1.shape} vs TF {tf_layer1.shape}")

                # Compare layer2 outputs
                pytorch_layer2 = activations['layer2_output']
                tf_layer2 = tf_intermediates[1]

                if pytorch_layer2.shape == tf_layer2.shape:
                    diff = np.abs(pytorch_layer2 - tf_layer2)
                    print(f"Layer2 difference - mean: {np.mean(diff):.8f}, max: {np.max(diff):.8f}")
                    print(f"Layer2 correlation: {np.corrcoef(pytorch_layer2.flatten(), tf_layer2.flatten())[0,1]:.8f}")
                else:
                    print(f"Layer2 shape mismatch: PyTorch {pytorch_layer2.shape} vs TF {tf_layer2.shape}")

                # Compare fc1 outputs
                pytorch_fc1 = activations['fc1_output']
                tf_fc1 = tf_intermediates[2]

                if pytorch_fc1.shape == tf_fc1.shape:
                    diff = np.abs(pytorch_fc1 - tf_fc1)
                    print(f"FC1 difference - mean: {np.mean(diff):.8f}, max: {np.max(diff):.8f}")
                    print(f"FC1 correlation: {np.corrcoef(pytorch_fc1.flatten(), tf_fc1.flatten())[0,1]:.8f}")
                else:
                    print(f"FC1 shape mismatch: PyTorch {pytorch_fc1.shape} vs TF {tf_fc1.shape}")

                # Compare final outputs
                diff = np.abs(pytorch_final - tf_intermediates[3][0])
                print(f"Final output difference - mean: {np.mean(diff):.8f}, max: {np.max(diff):.8f}")
                print(f"Final output correlation: {np.corrcoef(pytorch_final.flatten(), tf_intermediates[3][0].flatten())[0,1]:.8f}")

            except Exception as e:
                print(f"Error getting TensorFlow intermediate activations: {e}")

    except Exception as e:
        print(f"Error loading TensorFlow model: {e}")

if __name__ == "__main__":
    compare_intermediate_activations()
