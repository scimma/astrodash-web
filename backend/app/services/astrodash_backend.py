import os
import pickle
import numpy as np
import json
import logging
from astropy.time import Time
from scipy.signal import medfilt, argrelmax
from scipy.interpolate import splrep, splev
from scipy.stats import chisquare, pearsonr
from collections import OrderedDict
from urllib.error import URLError

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("astrodash_backend")

# PyTorch Model
class AstroDashPyTorchNet(nn.Module):
    """A PyTorch implementation of the AstroDash CNN."""
    def __init__(self, n_types, im_width=32):
        super(AstroDashPyTorchNet, self).__init__()
        self.im_width = im_width
        logger.info(f"Initializing AstroDashPyTorchNet with {n_types} types and image width {im_width}.")
        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # After two max-pooling layers, a 32x32 image becomes 8x8
        pooled_size = (self.im_width // 4)
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(64 * pooled_size * pooled_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, n_types),
        )

    def forward(self, x):
        x = x.view(-1, 1, self.im_width, self.im_width)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return F.softmax(x, dim=1)


def get_training_parameters():
    """Load training parameters from the model directory"""
    services_dir = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.join(services_dir, '..', '..')
    models_dir = os.path.join(backend_root, 'astrodash_models')
    logger.info("Loading training parameters from pickle file.")
    with open(os.path.join(models_dir, "zeroZ/training_params.pickle"), 'rb') as f:
        pars = pickle.load(f, encoding='latin1')
    logger.info("Training parameters loaded.")
    return pars


class AgeBinning:
    """Handle age binning for supernova classification"""
    def __init__(self, min_age, max_age, age_bin_size):
        self.min_age = min_age
        self.max_age = max_age
        self.age_bin_size = age_bin_size
        logger.debug(f"Initialized AgeBinning: min_age={min_age}, max_age={max_age}, bin_size={age_bin_size}")

    def age_bin(self, age):
        return int(round(age / self.age_bin_size)) - int(round(self.min_age / self.age_bin_size))

    def age_labels(self):
        age_labels = []
        age_bin_prev = 0
        age_label_min = self.min_age
        for age in np.arange(self.min_age, self.max_age, 0.5):
            age_bin = self.age_bin(age)
            if age_bin != age_bin_prev:
                age_label_max = int(round(age))
                age_labels.append(f"{int(age_label_min)} to {age_label_max}")
                age_label_min = age_label_max
            age_bin_prev = age_bin
        age_labels.append(f"{int(age_label_min)} to {int(self.max_age)}")
        return age_labels


class CreateLabels:
    """Create classification labels"""
    def __init__(self, n_types, min_age, max_age, age_bin_size, type_list):
        self.n_types = n_types
        self.age_binning = AgeBinning(min_age, max_age, age_bin_size)
        self.type_list = type_list
        logger.debug(f"CreateLabels initialized with {n_types} types.")

    def type_names_list(self):
        type_names_list = []
        for t_type in self.type_list:
            for age_label in self.age_binning.age_labels():
                type_names_list.append(f"{t_type}: {age_label}")
        logger.debug(f"Generated {len(type_names_list)} type names.")
        return np.array(type_names_list)


def normalise_spectrum(flux):
    if len(flux) == 0 or np.min(flux) == np.max(flux):
        logger.warning("Normalising spectrum: zero or constant flux array.")
        return np.zeros(len(flux))
    return (flux - np.min(flux)) / (np.max(flux) - np.min(flux))


def zero_non_overlap_part(array, min_index, max_index, outer_val=0.0):
    sliced_array = np.copy(array)
    sliced_array[0:min_index] = outer_val
    sliced_array[max_index:] = outer_val
    return sliced_array


def limit_wavelength_range(wave, flux, min_wave, max_wave):
    logger.debug(f"Limiting wavelength range: min_wave={min_wave}, max_wave={max_wave}")
    if min_wave is not None:
        min_idx = (np.abs(wave - min_wave)).argmin()
        flux[:min_idx] = 0
    if max_wave is not None:
        max_idx = (np.abs(wave - max_wave)).argmin()
        flux[max_idx:] = 0
    return flux

# Spectrum Processor strictly for Dash Model
class SpectrumProcessorML:
    """Process spectrum data for classification"""
    def __init__(self, w0, w1, nw):
        self.w0 = w0
        self.w1 = w1
        self.nw = nw
        self.num_spline_points = 13
        logger.info(f"Initialized SpectrumProcessor: w0={w0}, w1={w1}, nw={nw}")

    def process_spectrum_data(self, wave, flux, z, smooth, min_wave, max_wave):
        logger.info(f"Processing spectrum data with {len(wave)} points.")
        flux = normalise_spectrum(flux)
        flux = limit_wavelength_range(wave, flux, min_wave, max_wave)
        if smooth > 0:
            wavelength_density = (np.max(wave) - np.min(wave)) / len(wave)
            w_density = (self.w1 - self.w0) / self.nw
            filter_size = int(w_density / wavelength_density * smooth / 2) * 2 + 1
            if filter_size >= 3:
                flux = medfilt(flux, kernel_size=filter_size)
        wave_deredshifted = wave / (1 + z)
        if len(wave_deredshifted) < 2:
            logger.error("Spectrum is out of classification range after deredshifting.")
            raise ValueError(f"Spectrum is out of classification range.")
        binned_wave, binned_flux, min_idx, max_idx = self.log_wavelength_binning(wave_deredshifted, flux)
        new_flux, _ = self.continuum_removal(binned_wave, binned_flux, min_idx, max_idx)
        mean_zero_flux = self.mean_zero(new_flux, min_idx, max_idx)
        apodized_flux = self.apodize(mean_zero_flux, min_idx, max_idx)
        flux_norm = normalise_spectrum(apodized_flux)
        flux_norm = zero_non_overlap_part(flux_norm, min_idx, max_idx, outer_val=0.5)
        logger.info(f"Spectrum processing complete. Output points: {len(flux_norm)}")
        return flux_norm, min_idx, max_idx, z

    def log_wavelength_binning(self, wave, flux):
        dwlog = np.log(self.w1 / self.w0) / self.nw
        wlog = self.w0 * np.exp(np.arange(0, self.nw) * dwlog)
        binned_flux = np.interp(wlog, wave, flux, left=0, right=0)
        non_zero_indices = np.where(binned_flux != 0)[0]
        min_index = non_zero_indices[0] if len(non_zero_indices) > 0 else 0
        max_index = non_zero_indices[-1] if len(non_zero_indices) > 0 else self.nw -1
        logger.debug(f"Binned wavelength: min={wlog.min()}, max={wlog.max()}, nonzero range=({min_index}, {max_index})")
        return wlog, binned_flux, min_index, max_index

    def continuum_removal(self, wave, flux, min_idx, max_idx):
        wave_region, flux_region = wave[min_idx:max_idx+1], flux[min_idx:max_idx+1]
        if len(wave_region) > self.num_spline_points:
            indices = np.linspace(0, len(wave_region)-1, self.num_spline_points, dtype=int)
            spline = splrep(wave_region[indices], flux_region[indices], k=3)
            continuum = splev(wave_region, spline)
        else:
            continuum = np.mean(flux_region)
        full_continuum = np.zeros_like(flux)
        full_continuum[min_idx:max_idx+1] = continuum
        return flux - full_continuum, full_continuum

    def mean_zero(self, flux, min_idx, max_idx):
        flux_out = np.copy(flux)
        flux_out[0:min_idx] = flux_out[min_idx]
        flux_out[max_idx:] = flux_out[min_idx]
        return flux_out - flux_out[min_idx]

    def apodize(self, flux, min_idx, max_idx):
        apodized = np.copy(flux)
        edge_width = min(50, (max_idx - min_idx) // 4)
        if edge_width > 0:
            for i in range(edge_width):
                factor = 0.5 * (1 + np.cos(np.pi * i / edge_width))
                if min_idx + i < len(apodized): apodized[min_idx + i] *= factor
                if max_idx - i >= 0: apodized[max_idx - i] *= factor
        return apodized


class LoadInputSpectra:
    """Load and process input spectra for classification"""
    def __init__(self, file_path_or_data, z, smooth, pars, min_wave, max_wave):
        self.nw = pars['nw']
        n_types, w0, w1, min_age, max_age, age_bin_size, type_list = (
            pars['nTypes'], pars['w0'], pars['w1'], pars['minAge'],
            pars['maxAge'], pars['ageBinSize'], pars['typeList']
        )
        self.type_names_list = CreateLabels(n_types, min_age, max_age, age_bin_size, type_list).type_names_list()
        self.n_bins = len(self.type_names_list)
        processor = SpectrumProcessorML(w0, w1, self.nw)
        logger.info(f"Loading and processing input spectra. z={z}, smooth={smooth}, min_wave={min_wave}, max_wave={max_wave}")
        if isinstance(file_path_or_data, str):
            data = np.loadtxt(file_path_or_data)
            wave, flux = data[:, 0], data[:, 1]
        else: # It's an object with x and y attributes
            wave, flux = file_path_or_data.x, file_path_or_data.y
        self.flux, self.min_index, self.max_index, self.z = processor.process_spectrum_data(
            np.array(wave), np.array(flux), z, smooth, min_wave, max_wave
        )

    def input_spectra(self):
        logger.debug("Converting processed flux to torch tensor for model input.")
        input_images = torch.from_numpy(self.flux).float().reshape(1, -1)
        return input_images, [self.z], self.type_names_list, int(self.nw), self.n_bins, [(self.min_index, self.max_index)]


class BestTypesListSingleRedshift:
    """Get best classification types using the PyTorch model."""
    def __init__(self, model_path, input_images, type_names_list, nw, n_bins):
        self.type_names_list = np.array(type_names_list)
        logger.info(f"Loading PyTorch model from {model_path}")

        # Load the saved model state to determine the actual number of output classes
        saved_state_dict = torch.load(model_path)
        actual_n_types = saved_state_dict['classifier.3.weight'].shape[0]
        logger.info(f"Saved model has {actual_n_types} output classes, but we need {n_bins} classes")

        # Create model with the actual number of output classes from the saved model
        model = AstroDashPyTorchNet(actual_n_types)
        model.load_state_dict(saved_state_dict)
        model.eval()
        logger.info("Model loaded and set to eval mode.")

        with torch.no_grad():
            outputs = model(input_images)

        self.best_types = []
        self.softmax_ordered = []
        for i in range(outputs.shape[0]):
            # Only use the first n_bins outputs (corresponding to actual galaxy types)
            # The model may have been trained with more classes but only the first n_bins are valid
            softmax = outputs[i].numpy()[:n_bins]
            best_types, _, softmax_ordered = self.create_list(softmax)
            self.best_types.append(best_types)
            self.softmax_ordered.append(softmax_ordered)
        logger.info("Classification inference complete.")

    def create_list(self, softmax):
        idx = np.argsort(softmax)[::-1]
        best_types = self.type_names_list[idx]
        return best_types, idx, softmax[idx]


# Functions to be used by other services

def classification_split(classification_string):
    parts = classification_string.split(': ')
    return "", parts[0], parts[1]


def combined_prob(best_match_list):
    prev_name, age, _ = best_match_list[0]
    prob_initial = float(best_match_list[0][2])
    best_name, prob_total = prev_name, 0.0
    prev_broad_type = prev_name[:2]
    ages_list = [int(v) for v in age.split(' to ')]
    prob_possible, ages_list_possible = 0., []
    prob_possible2, ages_list_possible2 = 0., []
    for i, (name, age, prob) in enumerate(best_match_list[:10]):
        min_age, max_age = map(int, age.split(' to '))
        broad_type = "Ib" if "IIb" in name else name[:2]
        if name == prev_name:
            if prob_possible == 0:
                if min_age in ages_list or max_age in ages_list:
                    prob_total += float(prob)
                    ages_list.extend([min_age, max_age])
                else:
                    prob_possible = float(prob)
                    ages_list_possible = [min_age, max_age]
        elif broad_type == prev_broad_type:
            if prob_possible == 0:
                if i <= 1: best_name = broad_type
                prob_total += float(prob)
                ages_list.extend([min_age, max_age])
    if prob_total < prob_initial: prob_total = prob_initial
    best_age = f'{min(ages_list)} to {max(ages_list)}'
    reliable_flag = prob_total > prob_initial
    logger.debug(f"Combined probability: best_type={best_name}, best_age={best_age}, prob_total={prob_total}, reliable={reliable_flag}")
    return best_name, best_age, round(prob_total, 4), reliable_flag

def load_template_spectrum(sn_type, age_bin, npz_path, pars):
    logger.info(f"Loading template spectrum for SN type {sn_type}, age bin {age_bin}")
    data = np.load(npz_path, allow_pickle=True)
    snTemplates_raw = data['snTemplates'].item() if 'snTemplates' in data else data['arr_0'].item()
    snTemplates = {str(k): v for k, v in snTemplates_raw.items()}
    if not isinstance(snTemplates[sn_type], dict):
        snTemplates[sn_type] = dict(snTemplates[sn_type])
    if age_bin not in snTemplates[sn_type].keys():
        logger.error(f"Age bin '{age_bin}' not found for SN type '{sn_type}'.")
        raise ValueError(f"Age bin '{age_bin}' not found for SN type '{sn_type}'.")
    snInfo = snTemplates[sn_type][age_bin].get('snInfo', None)
    if not isinstance(snInfo, np.ndarray) or snInfo.shape[0] == 0:
        logger.error(f"No template spectrum available for SN type '{sn_type}' and age bin '{age_bin}'.")
        raise ValueError(f"No template spectrum available for SN type '{sn_type}' and age bin '{age_bin}'.")
    template = snInfo[0]  # placeholder for now
    wave = template[0]
    flux = template[1]
    logger.info(f"Template spectrum loaded for {sn_type} / {age_bin}")
    return wave, flux

def get_valid_sn_types_and_age_bins(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    snTemplates = data['snTemplates'].item()
    valid = {}
    for sn_type, age_bins in snTemplates.items():
        valid_bins = []
        for age_bin, entry in age_bins.items():
            snInfo = entry.get('snInfo', None)
            if (
                isinstance(snInfo, np.ndarray) and
                snInfo.shape and
                len(snInfo.shape) == 2 and
                snInfo.shape[0] > 0 and
                snInfo.shape[1] == 4
            ):
                valid_bins.append(age_bin)
        if valid_bins:
            valid[sn_type] = valid_bins
    logger.info(f"Valid SN types and age bins loaded: {list(valid.keys())}")
    return valid


__all__ = [
    'get_training_parameters', 'BestTypesListSingleRedshift', 'LoadInputSpectra',
    'classification_split', 'combined_prob', 'RlapCalc', 'normalise_spectrum',
    'AstroDashPyTorchNet', 'AgeBinning', 'CreateLabels', 'LoadInputSpectra',
    'BestTypesListSingleRedshift', 'RlapCalc', 'normalise_spectrum', 'get_valid_sn_types_and_age_bins'
]
