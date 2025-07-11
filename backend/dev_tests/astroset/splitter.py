"""
splitter.py
-----------
Functions for splitting the dataset into train and test sets for the Astrodash training set pipeline.
"""

import numpy as np

def split_train_test(dataset_dict, train_fraction, by='spectrum'):
    """
    Split the dataset into train and test sets by spectrum (shuffle before splitting).
    Args:
        dataset_dict (dict): Dataset dict with keys 'images', 'labels', 'filenames', 'type_names'.
        train_fraction (float): Fraction of data to use for training.
        by (str): 'spectrum' (default; 'file' not implemented).
    Returns:
        tuple: (train_dict, test_dict), each a dataset dict.
    """
    n = len(dataset_dict['labels'])
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_train = int(train_fraction * n)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    def subset(d, idx):
        return [d[i] if isinstance(d, list) else d[idx] for i in idx] if isinstance(d, list) else d[idx]
    train_dict = {k: subset(v, train_idx) for k, v in dataset_dict.items()}
    test_dict = {k: subset(v, test_idx) for k, v in dataset_dict.items()}
    return train_dict, test_dict
