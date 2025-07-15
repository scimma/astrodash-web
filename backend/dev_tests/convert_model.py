import os
import pickle
import numpy as np
import torch
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse

# --- PyTorch Model Definition (must match the one in astrodash_backend.py) ---
class AstroDashPyTorchNet(torch.nn.Module):
    def __init__(self, n_types, im_width=32):
        super(AstroDashPyTorchNet, self).__init__()
        self.im_width = im_width
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=5, padding='same'),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding='same'),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        pooled_size = im_width // 4
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(64 * pooled_size * pooled_size, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, n_types),
        )

    def forward(self, x):
        x = x.view(-1, 1, self.im_width, self.im_width)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.nn.functional.softmax(x, dim=1)


# --- TensorFlow Model Definition (from the original AstroDash code) ---
def convnet_variables(im_width, im_width_reduc, n, n_types):
    """Rebuilds the original TensorFlow graph to extract variables."""
    x = tf.placeholder(tf.float32, shape=[None, n])
    y_ = tf.placeholder(tf.float32, shape=[None, n_types])
    x_image = tf.reshape(x, [-1, im_width, im_width, 1])

    def _weight_variable(shape, name):
        return tf.get_variable(name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.1))

    def _bias_variable(shape, name):
        return tf.get_variable(name, shape=shape, initializer=tf.constant_initializer(0.1))

    with tf.variable_scope("layer1"):
        W_conv1 = _weight_variable([5, 5, 1, 32], "W_conv1")
        b_conv1 = _bias_variable([32], "b_conv1")
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope("layer2"):
        W_conv2 = _weight_variable([5, 5, 32, 64], "W_conv2")
        b_conv2 = _bias_variable([64], "b_conv2")
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Note: The original code had confusing, unused layers. This is the actual architecture used.
        h_pool_flat = tf.reshape(h_pool2, [-1, (im_width // 4) * (im_width // 4) * 64])
        W_fc1 = _weight_variable([(im_width // 4) * (im_width // 4) * 64, 1024], "W_fc1")
        b_fc1 = _bias_variable([1024], "b_fc1")
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

    with tf.variable_scope("readout"):
        keep_prob = tf.placeholder(tf.float32)
        h_fc_drop = tf.nn.dropout(h_fc1, keep_prob)
        W_fc2 = _weight_variable([1024, n_types], "W_fc2")
        b_fc2 = _bias_variable([n_types], "b_fc2")
        y_conv = tf.nn.softmax(tf.matmul(h_fc_drop, W_fc2) + b_fc2)

    return y_conv # We only need the graph structure for variable loading

def main():
    """Main conversion function."""
    print("Starting TensorFlow to PyTorch model conversion...")

    # --- Configuration ---
    parser = argparse.ArgumentParser(description='Convert TensorFlow AstroDash model to PyTorch')
    parser.add_argument('--model_dir', type=str, default='zeroZ', help='Model subdirectory to convert (zeroZ or agnosticZ)')
    args = parser.parse_args()

    base_path = os.path.join(os.path.dirname(__file__), '..', 'astrodash_models', args.model_dir)
    tf_model_path = os.path.join(base_path, 'tensorflow_model.ckpt')
    pytorch_model_path = os.path.join(base_path, 'pytorch_model.pth')
    params_path = os.path.join(base_path, 'training_params.pickle')

    if not os.path.exists(tf_model_path + ".index"):
        print(f"Error: TensorFlow model not found at {tf_model_path}")
        print("Please ensure the model data is downloaded and placed correctly.")
        return

    with open(params_path, 'rb') as f:
        pars = pickle.load(f, encoding='latin1')

    im_width = int(np.sqrt(pars['nw']))

    # --- Conversion using checkpoint reader only ---
    import tensorflow as tf
    reader = tf.compat.v1.train.NewCheckpointReader(tf_model_path)
    var_to_shape_map = reader.get_variable_to_shape_map()

    print("Checkpoint variable shapes:")
    for k, v in var_to_shape_map.items():
        print(f"{k}: {v}")

    # Always determine n_types from checkpoint: find all variables with shape (1024, N) and use the largest N
    max_n_types = 0
    for k, v in var_to_shape_map.items():
        if isinstance(v, (tuple, list)) and len(v) == 2 and v[0] == 1024:
            if v[1] > max_n_types:
                max_n_types = v[1]
    if max_n_types > 0:
        n_types = max_n_types
        print(f"[INFO] Forcing n_types={n_types} to match checkpoint output size. Downstream code should use only the first len(typeNamesList) outputs for user-facing results, as in legacy AstroDash.")
        if n_types != pars['nTypes']:
            print(f"[WARNING] training_params.pickle nTypes={pars['nTypes']} but checkpoint output is {n_types}. This is expected for legacy models.")
    else:
        n_types = pars['nTypes']
        print(f"[INFO] Using n_types={n_types} from parameter file (could not infer from checkpoint)")

    # Filter out optimizer variables and get only the main weights/biases
    main_vars = []
    for key in var_to_shape_map:
        if not any(x in key for x in ['Adam', 'beta1_power', 'beta2_power']):
            main_vars.append((key, reader.get_tensor(key)))

    # Sort by variable name to ensure consistent order (Variable, Variable_1, ...)
    main_vars_sorted = sorted(main_vars, key=lambda x: x[0])

    # Print variable names and shapes for sanity check
    for i, (k, v) in enumerate(main_vars_sorted):
        print(f"{i}: {k} shape={v.shape}")

    # Helper to find variable by shape
    def find_var_by_shape(vars_list, shape):
        for i, (k, v) in enumerate(vars_list):
            if v.shape == shape:
                return v
        print("Available shapes in checkpoint:")
        for i, (k, v) in enumerate(vars_list):
            print(f"{k}: {v.shape}")
        raise ValueError(f"No variable with shape {shape}")

    # Assign variables to PyTorch model using shape-based mapping
    pytorch_model = AstroDashPyTorchNet(n_types=n_types, im_width=im_width)
    pytorch_state_dict = pytorch_model.state_dict()
    pytorch_state_dict['features.0.weight'].copy_(
        torch.from_numpy(np.transpose(find_var_by_shape(main_vars_sorted, (5, 5, 1, 32)), (3, 2, 0, 1))))
    pytorch_state_dict['features.0.bias'].copy_(
        torch.from_numpy(find_var_by_shape(main_vars_sorted, (32,))))
    pytorch_state_dict['features.3.weight'].copy_(
        torch.from_numpy(np.transpose(find_var_by_shape(main_vars_sorted, (5, 5, 32, 64)), (3, 2, 0, 1))))
    pytorch_state_dict['features.3.bias'].copy_(
        torch.from_numpy(find_var_by_shape(main_vars_sorted, (64,))))
    pytorch_state_dict['classifier.0.weight'].copy_(
        torch.from_numpy(find_var_by_shape(main_vars_sorted, (4096, 1024)).T))
    pytorch_state_dict['classifier.0.bias'].copy_(
        torch.from_numpy(find_var_by_shape(main_vars_sorted, (1024,))))
    pytorch_state_dict['classifier.3.weight'].copy_(
        torch.from_numpy(find_var_by_shape(main_vars_sorted, (1024, n_types)).T))
    pytorch_state_dict['classifier.3.bias'].copy_(
        torch.from_numpy(find_var_by_shape(main_vars_sorted, (n_types,))))

    print("- Mapped all variables by shape to PyTorch model.")

    # Save the new PyTorch model
    torch.save(pytorch_state_dict, pytorch_model_path)
    print(f"\nSuccessfully converted and saved PyTorch model to:\n{pytorch_model_path}")

if __name__ == '__main__':
    main()
