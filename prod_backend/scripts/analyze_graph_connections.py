import sys
import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def analyze_graph_connections():
    """Analyze the exact TensorFlow graph connections to understand the architecture."""
    tf_model_path = "backend/dev_tests/../astrodash_models/zeroZ/tensorflow_model.ckpt"

    # Load the meta graph
    saver = tf.train.import_meta_graph(tf_model_path + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, tf_model_path)
        graph = tf.get_default_graph()

        print("=== Detailed Graph Connection Analysis ===")

        # Get input and output tensors
        x = graph.get_tensor_by_name("Placeholder:0")
        y_conv = graph.get_tensor_by_name("Softmax:0")

        print(f"Input: {x.name}, shape: {x.shape}")
        print(f"Output: {y_conv.name}, shape: {y_conv.shape}")

        # Trace the forward pass step by step
        print("\n=== Forward Pass Trace ===")

        # Start from output and trace backwards
        current_tensor = y_conv
        depth = 0
        max_depth = 30
        visited = set()

        def trace_backwards(tensor, depth, path=""):
            if depth > max_depth or tensor.name in visited:
                return

            visited.add(tensor.name)
            op = tensor.op

            print(f"{'  ' * depth}Depth {depth}: {op.name} ({op.type})")
            print(f"{'  ' * depth}  Tensor: {tensor.name}")
            print(f"{'  ' * depth}  Shape: {tensor.shape}")
            print(f"{'  ' * depth}  Path: {path}")

            # Get inputs to this operation
            inputs = op.inputs
            if inputs:
                print(f"{'  ' * depth}  Inputs:")
                for i, inp in enumerate(inputs):
                    print(f"{'  ' * depth}    {i}: {inp.name} (from {inp.op.name})")
                    trace_backwards(inp, depth + 1, f"{path} -> {op.name}")
            else:
                print(f"{'  ' * depth}  No inputs (leaf node)")
            print()

        trace_backwards(y_conv, 0, "output")

        # Also trace forward from input
        print("\n=== Forward Pass from Input ===")
        visited_forward = set()

        def trace_forward(tensor, depth, path=""):
            if depth > max_depth or tensor.name in visited_forward:
                return

            visited_forward.add(tensor.name)
            op = tensor.op

            print(f"{'  ' * depth}Depth {depth}: {op.name} ({op.type})")
            print(f"{'  ' * depth}  Tensor: {tensor.name}")
            print(f"{'  ' * depth}  Shape: {tensor.shape}")
            print(f"{'  ' * depth}  Path: {path}")

            # Get consumers of this tensor
            consumers = tensor.consumers()
            if consumers:
                print(f"{'  ' * depth}  Consumers:")
                for i, consumer in enumerate(consumers):
                    print(f"{'  ' * depth}    {i}: {consumer.name} -> {consumer.outputs[0].name}")
                    trace_forward(consumer.outputs[0], depth + 1, f"{path} -> {op.name}")
            else:
                print(f"{'  ' * depth}  No consumers (output node)")
            print()

        trace_forward(x, 0, "input")

        # Analyze specific operations
        print("\n=== Operation Analysis ===")

        # Find all conv operations
        conv_ops = [op for op in graph.get_operations() if 'Conv2D' in op.name and not 'gradient' in op.name]
        print(f"Convolutional operations: {[op.name for op in conv_ops]}")

        # Find all matmul operations
        matmul_ops = [op for op in graph.get_operations() if 'MatMul' in op.name and not 'gradient' in op.name]
        print(f"Matrix multiplication operations: {[op.name for op in matmul_ops]}")

        # Find all add operations
        add_ops = [op for op in graph.get_operations() if 'add' in op.name and not 'gradient' in op.name]
        print(f"Add operations: {[op.name for op in add_ops]}")

        # Find all relu operations
        relu_ops = [op for op in graph.get_operations() if 'Relu' in op.name and not 'gradient' in op.name]
        print(f"ReLU operations: {[op.name for op in relu_ops]}")

        # Find all max pool operations
        pool_ops = [op for op in graph.get_operations() if 'MaxPool' in op.name and not 'gradient' in op.name]
        print(f"MaxPool operations: {[op.name for op in pool_ops]}")

        # Find all reshape operations
        reshape_ops = [op for op in graph.get_operations() if 'Reshape' in op.name and not 'gradient' in op.name]
        print(f"Reshape operations: {[op.name for op in reshape_ops]}")

        # Analyze the data flow
        print("\n=== Data Flow Analysis ===")

        # Check if there are any branching or parallel paths
        print("Looking for branching patterns...")

        # Find tensors with multiple consumers
        multi_consumer_tensors = []
        for op in graph.get_operations():
            for output in op.outputs:
                consumers = output.consumers()
                if len(consumers) > 1:
                    multi_consumer_tensors.append((output.name, [c.name for c in consumers]))

        if multi_consumer_tensors:
            print("Tensors with multiple consumers (potential branching):")
            for tensor_name, consumers in multi_consumer_tensors:
                print(f"  {tensor_name} -> {consumers}")
        else:
            print("No branching detected - linear flow")

if __name__ == "__main__":
    analyze_graph_connections()
