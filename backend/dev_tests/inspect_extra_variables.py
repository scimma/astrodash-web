import sys
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def inspect_extra_variables():
    """Inspect the extra TensorFlow variables to understand the model architecture."""
    tf_model_path = "backend/dev_tests/../astrodash_models/zeroZ/tensorflow_model.ckpt"

    reader = tf.train.NewCheckpointReader(tf_model_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    # Filter out optimizer variables
    main_vars = {}
    for key in var_to_shape_map:
        if not any(x in key for x in ['Adam', 'beta1_power', 'beta2_power']):
            main_vars[key] = reader.get_tensor(key)

    print("=== All TensorFlow Variables ===")
    for name, tensor in sorted(main_vars.items()):
        print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
        print(f"  Min: {np.min(tensor):.6f}, Max: {np.max(tensor):.6f}, Mean: {np.mean(tensor):.6f}")
        print()

    # Check if there are additional conv layers
    print("=== Looking for Additional Layers ===")
    conv_vars = [name for name, tensor in main_vars.items() if len(tensor.shape) == 4]
    fc_vars = [name for name, tensor in main_vars.items() if len(tensor.shape) == 2]
    bias_vars = [name for name, tensor in main_vars.items() if len(tensor.shape) == 1]

    print(f"Convolutional variables: {conv_vars}")
    print(f"Fully connected variables: {fc_vars}")
    print(f"Bias variables: {bias_vars}")

    # Check if Variable_6 and Variable_7 are additional conv layers
    if 'Variable_6' in main_vars and 'Variable_7' in main_vars:
        print("\n=== Variable_6 and Variable_7 Analysis ===")
        var6 = main_vars['Variable_6']
        var7 = main_vars['Variable_7']
        print(f"Variable_6 shape: {var6.shape} (looks like conv layer)")
        print(f"Variable_7 shape: {var7.shape} (looks like bias)")

        if len(var6.shape) == 4 and len(var7.shape) == 1:
            print("These appear to be an additional convolutional layer!")

    # Check if Variable_8 and Variable_9 are additional FC layers
    if 'Variable_8' in main_vars and 'Variable_9' in main_vars:
        print("\n=== Variable_8 and Variable_9 Analysis ===")
        var8 = main_vars['Variable_8']
        var9 = main_vars['Variable_9']
        print(f"Variable_8 shape: {var8.shape} (looks like FC layer)")
        print(f"Variable_9 shape: {var9.shape} (looks like bias)")

        if len(var8.shape) == 2 and len(var9.shape) == 1:
            print("These appear to be an additional fully connected layer!")

if __name__ == "__main__":
    inspect_extra_variables()
