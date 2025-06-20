import tensorflow as tf
import pickle

checkpoint_path = "astrodash_data/models_v06/models/zeroZ/tensorflow_model.ckpt"
reader = tf.compat.v1.train.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print(key)

paths = [
    'astrodash_data/models_v06/models/zeroZ/training_params.pickle',
    'astrodash_data/models_v06/models/agnosticZ/training_params.pickle',
    'astrodash_data/models_v06/models/zeroZ_classifyHost/training_params.pickle',
]

for path in paths:
    try:
        with open(path, 'rb') as f:
            pars = pickle.load(f, encoding='latin1')
        print(f"{path}: nTypes = {pars.get('nTypes', 'NOT FOUND')}")
    except Exception as e:
        print(f"{path}: ERROR - {e}")
