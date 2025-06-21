import tensorflow as tf
import pickle

checkpoint_path = "astrodash_data/models_v06/models/zeroZ/tensorflow_model.ckpt"
reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print(key)

def inspect_pickle_file(file_path):
    """Prints the contents of a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            print(f"Successfully loaded {file_path}")
            print("Contents:")
            print(data)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # List of pickle files to inspect
    pickle_files = [
        'backend/astrodash_models/zeroZ/training_params.pickle',
        'backend/astrodash_models/agnosticZ/training_params.pickle',
    ]

    for p_file in pickle_files:
        inspect_pickle_file(p_file)
        print("-" * 30)
