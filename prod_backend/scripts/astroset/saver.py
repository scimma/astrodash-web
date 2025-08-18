"""
saver.py
--------
Functions for saving and loading dataset arrays for the Astrodash training set pipeline.
"""

import numpy as np
import pickle
import os

def save_dataset(dataset_dict, outdir, prefix):
    """
    Save dataset arrays as .npy files in the output directory.
    Args:
        dataset_dict (dict): Dataset dict with keys 'images', 'labels', 'filenames', 'type_names'.
        outdir (str): Output directory.
        prefix (str): Prefix for saved files (e.g., 'train' or 'test').
    """
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"{prefix}_images.npy"), dataset_dict['images'])
    np.save(os.path.join(outdir, f"{prefix}_labels.npy"), dataset_dict['labels'])
    with open(os.path.join(outdir, f"{prefix}_filenames.pkl"), 'wb') as f:
        pickle.dump(dataset_dict['filenames'], f)
    with open(os.path.join(outdir, f"{prefix}_type_names.pkl"), 'wb') as f:
        pickle.dump(dataset_dict['type_names'], f)

def load_dataset(outdir, prefix):
    """
    Load dataset arrays from .npy files in the output directory.
    Args:
        outdir (str): Output directory.
        prefix (str): Prefix for saved files.
    Returns:
        dict: Loaded dataset dict.
    """
    images = np.load(os.path.join(outdir, f"{prefix}_images.npy"))
    labels = np.load(os.path.join(outdir, f"{prefix}_labels.npy"))
    with open(os.path.join(outdir, f"{prefix}_filenames.pkl"), 'rb') as f:
        filenames = pickle.load(f)
    with open(os.path.join(outdir, f"{prefix}_type_names.pkl"), 'rb') as f:
        type_names = pickle.load(f)
    return {'images': images, 'labels': labels, 'filenames': filenames, 'type_names': type_names}
