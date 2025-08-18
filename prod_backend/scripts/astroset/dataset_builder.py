"""
dataset_builder.py
------------------
Functions for building dataset arrays for the Astrodash training set pipeline.
"""

import numpy as np
from interpolator import interpolate_spectrum
from labeler import assign_label

def build_dataset(spectra_list, config):
    """
    Build arrays for images, labels, filenames, and type names from parsed spectra.
    Args:
        spectra_list (list of dict): List of parsed spectra dicts.
        config (dict): Configuration dictionary.
    Returns:
        dict: {'images': np.ndarray, 'labels': np.ndarray, 'filenames': list, 'type_names': list}
    """
    min_age = config['minAge']
    max_age = config['maxAge']
    age_bin_size = config['ageBinSize']
    # Filter spectra to only those within the valid age range (Astrodash behavior)
    filtered_spectra = [spec for spec in spectra_list if min_age <= spec['age'] <= max_age]
    print(f"Filtered spectra: {len(filtered_spectra)} out of {len(spectra_list)}")

    images = []
    labels = []
    filenames = []
    type_names = []
    for spec in filtered_spectra:
        interp_flux = interpolate_spectrum(spec['wavelength'], spec['flux'], config)
        label = assign_label(spec['type'], spec['age'], config)
        images.append(interp_flux)
        labels.append(label)
        filenames.append(spec['filename'])
        # Use Astrodash/webapp age binning and display logic
        age_bin = int(round(spec['age'] / age_bin_size)) - int(round(min_age / age_bin_size))
        bin_left = min_age + age_bin * age_bin_size
        bin_right = bin_left + age_bin_size
        type_names.append(f"{spec['type']}: {int(bin_left)} to {int(bin_right)}")

    images = np.stack(images)
    labels = np.array(labels, dtype=np.int32)

    print(f"Label range: {labels.min()} to {labels.max()} ({len(np.unique(labels))} unique labels)")

    return {'images': images, 'labels': labels, 'filenames': filenames, 'type_names': type_names}
