import sys
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def debug_tf_variables():
    """Debug TensorFlow variables to understand which ones are actually used."""
    tf_model_path = "backend/dev_tests/../astrodash_models/zeroZ/tensorflow_model.ckpt"

    # Load the meta graph
    saver = tf.train.import_meta_graph(tf_model_path + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, tf_model_path)
        graph = tf.get_default_graph()

        print("=== Checking Variable Usage ===")

        # Get all variables
        all_vars = tf.global_variables()
        print(f"Total variables: {len(all_vars)}")

        # Check which variables are actually used in the forward pass
        x = graph.get_tensor_by_name("Placeholder:0")
        y_conv = graph.get_tensor_by_name("Softmax:0")

        # Get all operations in the path from input to output
        all_ops = graph.get_operations()

        # Find all variable read operations
        var_read_ops = [op for op in all_ops if '/read' in op.name]
        print(f"\nVariable read operations: {[op.name for op in var_read_ops]}")

        # Check which variables are actually read
        used_vars = set()
        for op in var_read_ops:
            var_name = op.name.replace('/read', '')
            used_vars.add(var_name)

        print(f"\nVariables actually used in forward pass: {sorted(used_vars)}")

        # Check which variables are NOT used
        all_var_names = {var.name for var in all_vars}
        unused_vars = all_var_names - used_vars
        print(f"\nVariables NOT used in forward pass: {sorted(unused_vars)}")

        # Let's also check the actual variable shapes and see if they make sense
        print("\n=== Variable Shapes Analysis ===")
        reader = tf.train.NewCheckpointReader(tf_model_path)
        var_to_shape_map = reader.get_variable_to_shape_map()

        for var_name in sorted(var_to_shape_map.keys()):
            if not any(x in var_name for x in ['Adam', 'beta1_power', 'beta2_power']):
                shape = var_to_shape_map[var_name]
                is_used = var_name in used_vars
                status = "✅ USED" if is_used else "❌ UNUSED"
                print(f"{status} {var_name}: {shape}")

        # Let's also check if there are any operations that use the "unused" variables
        print("\n=== Checking for operations using 'unused' variables ===")
        for var_name in unused_vars:
            if not any(x in var_name for x in ['Adam', 'beta1_power', 'beta2_power']):
                # Look for operations that might use this variable
                related_ops = [op for op in all_ops if var_name in op.name]
                if related_ops:
                    print(f"Variable {var_name} has related operations: {[op.name for op in related_ops]}")

if __name__ == "__main__":
    debug_tf_variables()
