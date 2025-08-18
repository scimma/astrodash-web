"""
labeler.py
----------
Functions for age binning and label assignment for the Astrodash training set pipeline.
"""

def age_bin(age, config):
    """
    Compute the age bin index for a given age, using Astrodash logic.
    Args:
        age (float): The age value.
        config (dict): Configuration dictionary.
    Returns:
        int: Age bin index.
    """
    min_age = config['minAge']
    age_bin_size = config['ageBinSize']
    # Astrodash: round((age - min_age) / age_bin_size)
    return int(round((age - min_age) / age_bin_size))

def assign_label(type_name, age, config, host_type=None):
    """
    Assign an integer label for a (type, age) combination (SN-only), matching Astrodash.
    Args:
        type_name (str): SN type name.
        age (float): Age value.
        config (dict): Configuration dictionary.
        host_type (str, optional): Ignored for SN-only.
    Returns:
        int: Label index.
    """
    type_list = config['typeList']
    n_types = config['nTypes']
    min_age = config['minAge']
    age_bin_size = config['ageBinSize']
    # Astrodash: num_age_bins = round((maxAge - minAge) / ageBinSize) + 1
    num_age_bins = int(round((config['maxAge'] - min_age) / age_bin_size)) + 1
    type_idx = type_list.index(type_name)
    age_idx = age_bin(age, config)
    label = type_idx * num_age_bins + age_idx

    return label

def get_type_names_list(config, host_types=None):
    """
    Generate a list of all possible class/type names (SN-only).
    Args:
        config (dict): Configuration dictionary.
        host_types (list, optional): Ignored for SN-only.
    Returns:
        list of str: List of class/type names.
    """
    type_list = config['typeList']
    num_age_bins = age_bin(config['maxAge'] - 0.1, config) + 1
    age_labels = []
    min_age = config['minAge']
    age_bin_size = config['ageBinSize']
    for i in range(num_age_bins):
        age_min = int(min_age + i * age_bin_size)
        age_max = int(min_age + (i + 1) * age_bin_size)
        age_labels.append(f"{age_min} to {age_max}")
    names = []
    for t in type_list:
        for a in age_labels:
            names.append(f"{t}: {a}")

    return names
