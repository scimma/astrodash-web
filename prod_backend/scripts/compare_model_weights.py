import sys
import os
import numpy as np
import torch
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.dev_tests.convert_model import AstroDashPyTorchNet

def load_tensorflow_weights(model_path):
    """Load TensorFlow model weights."""
    reader = tf.train.NewCheckpointReader(model_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Filter out optimizer variables
    main_vars = {}
    for key in var_to_shape_map:
        if not any(x in key for x in ['Adam', 'beta1_power', 'beta2_power']):
            main_vars[key] = reader.get_tensor(key)

    return main_vars

def load_pytorch_weights(model_path):
    """Load PyTorch model weights."""
    return torch.load(model_path, map_location='cpu')

def compare_weights(tf_weights, pytorch_weights, layer_mapping):
    """Compare weights between TensorFlow and PyTorch models."""
    print("=== Weight Comparison ===")

    for tf_name, pytorch_name, transpose_info in layer_mapping:
        if tf_name in tf_weights and pytorch_name in pytorch_weights:
            tf_weight = tf_weights[tf_name]
            pytorch_weight = pytorch_weights[pytorch_name].numpy()

            # Apply transposition if needed
            if transpose_info:
                tf_weight = np.transpose(tf_weight, transpose_info)

            print(f"\n{tf_name} -> {pytorch_name}")
            print(f"  TF shape: {tf_weight.shape}")
            print(f"  PyTorch shape: {pytorch_weight.shape}")

            if tf_weight.shape == pytorch_weight.shape:
                diff = np.abs(tf_weight - pytorch_weight)
                print(f"  Mean absolute difference: {np.mean(diff):.8f}")
                print(f"  Max absolute difference: {np.max(diff):.8f}")
                print(f"  Correlation: {np.corrcoef(tf_weight.flatten(), pytorch_weight.flatten())[0,1]:.8f}")

                if np.max(diff) > 1e-6:
                    print(f"  ⚠️  WARNING: Significant difference detected!")
                else:
                    print(f"  ✅ Weights match!")
            else:
                print(f"  ❌ Shape mismatch!")
        else:
            print(f"\n❌ Missing weights: {tf_name} or {pytorch_name}")

def main():
    # Model paths
    tf_model_path = "backend/dev_tests/../astrodash_models/zeroZ/tensorflow_model.ckpt"
    pytorch_model_path = "backend/dev_tests/../astrodash_models/zeroZ/pytorch_model.pth"

    print("Loading TensorFlow weights...")
    tf_weights = load_tensorflow_weights(tf_model_path)

    print("Loading PyTorch weights...")
    pytorch_weights = load_pytorch_weights(pytorch_model_path)

    print(f"\nTensorFlow variables found: {list(tf_weights.keys())}")
    print(f"PyTorch variables found: {list(pytorch_weights.keys())}")

    # Define the expected mapping between TF and PyTorch weights
    # Format: (tf_variable_name, pytorch_variable_name, transpose_axes)
    layer_mapping = [
        # Conv1 layer
        ("Variable", "features.0.weight", (3, 2, 0, 1)),  # (5, 5, 1, 32) -> (32, 1, 5, 5)
        ("Variable_1", "features.0.bias", None),  # (32,) -> (32,)

        # Conv2 layer
        ("Variable_2", "features.3.weight", (3, 2, 0, 1)),  # (5, 5, 32, 64) -> (64, 32, 5, 5)
        ("Variable_3", "features.3.bias", None),  # (64,) -> (64,)

        # FC1 layer
        ("Variable_4", "classifier.0.weight", (1, 0)),  # (4096, 1024) -> (1024, 4096)
        ("Variable_5", "classifier.0.bias", None),  # (1024,) -> (1024,)

        # FC2 (output) layer
        ("Variable_10", "classifier.3.weight", (1, 0)),  # (1024, 306) -> (306, 1024)
        ("Variable_11", "classifier.3.bias", None),  # (306,) -> (306,)
    ]

    compare_weights(tf_weights, pytorch_weights, layer_mapping)

    # Also check for any extra variables
    print("\n=== Extra Variables ===")
    tf_vars = set(tf_weights.keys())
    pytorch_vars = set(pytorch_weights.keys())

    extra_tf = tf_vars - {m[0] for m in layer_mapping}
    extra_pytorch = pytorch_vars - {m[1] for m in layer_mapping}

    if extra_tf:
        print(f"Extra TensorFlow variables: {extra_tf}")
    if extra_pytorch:
        print(f"Extra PyTorch variables: {extra_pytorch}")

if __name__ == "__main__":
    main()
