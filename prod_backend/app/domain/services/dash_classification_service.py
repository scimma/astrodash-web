from typing import Any, Dict, Optional
import numpy as np
from app.infrastructure.ml.dash_utils import (
    load_training_parameters, LoadInputSpectra, BestTypesListSingleRedshift, combined_prob, classification_split
)
from app.domain.repositories.spectrum_repository import create_spectrum_template_handler
from app.shared.utils.helpers import (
    prepare_log_wavelength_and_templates, get_templates_for_type_age, get_nonzero_minmax, normalize_age_bin
)
from app.shared.utils.redshift import get_median_redshift
from app.config.logging import get_logger

class DashClassificationService:
    """
    Service to orchestrate Dash model classification, including preprocessing, inference, RLap, and result formatting.
    """
    def __init__(
        self,
        model_path_zero_z: str,
        model_path_agnostic_z: str,
        template_npz_path: str,
        logger: Optional[logging.Logger] = None
    ):
        self.model_path_zero_z = model_path_zero_z
        self.model_path_agnostic_z = model_path_agnostic_z
        self.template_npz_path = template_npz_path
        self.logger = logger or get_logger(__name__)
        self.pars = load_training_parameters(template_npz_path)

    async def classify(
        self,
        processed_data: Dict[str, Any],
        known_z: bool = False,
        calculate_rlap: bool = False
    ) -> Dict[str, Any]:
        """
        Classify a spectrum using the Dash model. Optionally calculate RLap and estimate redshift.
        """
        model_path = self.model_path_zero_z if known_z else self.model_path_agnostic_z
        self.logger.info(f"Starting classification. Using model: {model_path}")

        # Preprocess spectrum
        load_spectra = LoadInputSpectra(
            processed_data,
            z=processed_data.get('redshift', 0.0),
            smooth=0,
            pars=self.pars,
            min_wave=min(processed_data['x']),
            max_wave=max(processed_data['x'])
        )
        input_images, _, type_names_list, nw, n_bins, minmax_indexes = load_spectra.input_spectra()

        # Inference
        best_types_list = BestTypesListSingleRedshift(model_path, input_images, type_names_list, n_bins)
        matches = []
        if best_types_list.best_types:
            for i in range(len(best_types_list.best_types[0])):
                classification = best_types_list.best_types[0][i]
                probability = best_types_list.softmax_ordered[0][i]
                _, sn_name, sn_age = classification_split(classification)
                matches.append({
                    'type': sn_name, 'age': sn_age,
                    'probability': float(probability),
                    'redshift': processed_data.get('redshift', 0.0),
                    'rlap': None,  # Will be filled below
                    'reliable': False
                })
        if not matches:
            self.logger.warning("No matches found.")
            return {}

        # RLap calculation
        best_match = matches[0]
        estimated_redshift = None
        estimated_redshift_err = None
        if calculate_rlap:
            self.logger.info("Calculating RLap score using real template spectra.")
            log_wave, input_flux_log, snTemplates, dwlog, nw, w0, w1 = prepare_log_wavelength_and_templates(processed_data)
            best_match_idx = np.argmax([m['probability'] for m in matches])
            best_match = matches[best_match_idx]
            sn_type = best_match['type']
            age = best_match['age']
            age_norm = normalize_age_bin(age)
            template_fluxes, template_names, template_minmax_indexes = get_templates_for_type_age(snTemplates, sn_type, age_norm, log_wave)
            if template_fluxes:
                input_minmax_index = minmax_indexes[0] if minmax_indexes else get_nonzero_minmax(input_flux_log)
                est_z, _, _, est_z_err = await get_median_redshift(
                    input_flux_log, template_fluxes, nw, dwlog, input_minmax_index, template_minmax_indexes, template_names, outerVal=0.5
                )
                estimated_redshift = float(est_z) if est_z is not None else None
                estimated_redshift_err = float(est_z_err) if est_z_err is not None else None
                matches, best_match = self.compute_rlap_for_matches(
                    matches, best_match, log_wave, input_flux_log, template_fluxes, template_names, template_minmax_indexes, known_z
                )
            else:
                self.logger.warning(f"No valid templates found for sn_type '{sn_type}' and age '{age_norm}'")
                for m in matches:
                    m['rlap'] = "N/A"
                    m['rlap_warning'] = True
                best_match['rlap'] = "N/A"
                best_match['rlap_warning'] = True
        else:
            self.logger.info("RLAP calculation skipped as requested by user.")
            for m in matches:
                m['rlap'] = None
                m['rlap_warning'] = False
            best_match = matches[0] if matches else {}
            best_match['rlap'] = None
            best_match['rlap_warning'] = False

        # Probability combination
        best_match_list_for_prob = [[m['type'], m['age'], m['probability']] for m in matches]
        best_type, best_age, prob_total, reliable_flag = combined_prob(best_match_list_for_prob)
        best_match = {
            'type': best_type, 'age': best_age, 'probability': prob_total,
            'redshift': processed_data.get('redshift', 0.0),
            'rlap': best_match.get('rlap', None)
        }
        for m in matches:
            if m['type'] == best_type and m['age'] == best_age:
                m['reliable'] = reliable_flag
        # Attach estimated redshift if available
        if not known_z and estimated_redshift is not None:
            best_match['estimated_redshift'] = estimated_redshift
            best_match['estimated_redshift_error'] = estimated_redshift_err
            for m in matches:
                m['estimated_redshift'] = estimated_redshift
                m['estimated_redshift_error'] = estimated_redshift_err
        self.logger.info("Classification complete.")
        return {
            'best_matches': matches[:3],
            'best_match': best_match,
            'reliable_matches': reliable_flag
        }

    def compute_rlap_for_matches(
        self,
        matches: list,
        best_match: dict,
        log_wave: np.ndarray,
        input_flux_log: np.ndarray,
        template_fluxes: list,
        template_names: list,
        template_minmax_indexes: list,
        known_z: bool
    ) -> tuple:
        """
        Compute RLap for the best match and attach to matches and best_match dicts.
        """
        if not matches:
            return matches, best_match
        best_match_idx = np.argmax([m['probability'] for m in matches])
        best = matches[best_match_idx]
        input_minmax_index = get_nonzero_minmax(input_flux_log)
        rlap_label, used_redshift, rlap_warning = self._calculate_rlap_with_redshift(
            log_wave, input_flux_log, template_fluxes, template_names, template_minmax_indexes, input_minmax_index,
            redshift=best['redshift'] if known_z else None
        )
        for m in matches:
            m['rlap'] = rlap_label
            m['rlap_warning'] = rlap_warning
        best_match['rlap'] = rlap_label
        best_match['rlap_warning'] = rlap_warning
        return matches, best_match

    def _calculate_rlap_with_redshift(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        template_fluxes: list,
        template_names: list,
        template_minmax_indexes: list,
        input_minmax_index: tuple,
        redshift: Optional[float] = None
    ) -> tuple:
        """
        Calculate RLap score for the input spectrum using the best-match template.
        If redshift is not provided, estimate it using the templates.
        Returns (rlap_score, used_redshift, rlap_warning)
        """
        w0, w1, nw = self.pars['w0'], self.pars['w1'], self.pars['nw']
        dwlog = np.log(w1 / w0) / nw
        if redshift is None:
            self.logger.info("Estimating redshift for RLap calculation using best-match template(s).")
            est_redshift, _, _, _ = get_median_redshift(
                flux, template_fluxes, nw, dwlog, input_minmax_index, template_minmax_indexes, template_names
            )
            if est_redshift is None:
                self.logger.error("Redshift estimation failed. RLap will be calculated in observed frame.")
                est_redshift = 0.0
            redshift = est_redshift
        else:
            self.logger.info(f"Using provided redshift {redshift} for RLap calculation.")
        rest_wave = wave / (1 + redshift)
        rlap_label, rlap_warning = self._rlap_label(
            flux, template_fluxes, template_names, rest_wave, input_minmax_index, template_minmax_indexes, nw, dwlog
        )
        return rlap_label, redshift, rlap_warning

    def _rlap_label(
        self,
        input_flux: np.ndarray,
        template_fluxes: list,
        template_names: list,
        wave: np.ndarray,
        input_minmax_index: tuple,
        template_minmax_indexes: list,
        nw: int,
        dwlog: float
    ) -> tuple:
        """
        Compute RLap label and warning for the input and templates.
        """
        if not np.any(input_flux):
            return "No flux", True
        rlap_list = []
        for i in range(len(template_names)):
            r, lap, rlap, fom = self._rlap_score(
                input_flux, template_fluxes[i], template_minmax_indexes[i], nw, wave
            )
            rlap_list.append(rlap)
        rlap_mean = round(np.mean(rlap_list), 2)
        rlap_label = str(rlap_mean)
        rlap_warning = rlap_mean < 6
        return rlap_label, rlap_warning

    def _rlap_score(
        self,
        input_flux: np.ndarray,
        template_flux: np.ndarray,
        template_minmax_index: tuple,
        nw: int,
        wave: np.ndarray
    ) -> tuple:
        """
        Compute RLap score for a single template.
        """
        # Mean-zero both input and template
        from app.shared.utils.redshift import mean_zero_spectra
        input_flux_proc = mean_zero_spectra(input_flux, 0, nw - 1, nw)
        template_flux_proc = mean_zero_spectra(template_flux, template_minmax_index[0], template_minmax_index[1], nw)
        # Cross-correlation
        inputfourier = np.fft.fft(input_flux_proc)
        tempfourier = np.fft.fft(template_flux_proc)
        product = inputfourier * np.conj(tempfourier)
        xCorr = np.fft.fft(product)
        rmsInput = np.std(inputfourier)
        rmsTemp = np.std(tempfourier)
        xCorrNorm = (1. / (nw * rmsInput * rmsTemp)) * xCorr
        xCorrNormRearranged = np.concatenate((xCorrNorm[int(len(xCorrNorm) / 2):], xCorrNorm[0:int(len(xCorrNorm) / 2)]))
        crossCorr = np.correlate(input_flux_proc, template_flux_proc, mode='full')[::-1][int(nw / 2):int(nw + nw / 2)] / max(np.correlate(input_flux_proc, template_flux_proc, mode='full'))
        # Peaks
        from scipy.signal import argrelmax
        peakindexes = argrelmax(crossCorr)[0]
        ypeaks = [abs(crossCorr[i]) for i in peakindexes]
        arr = list(zip(peakindexes, ypeaks))
        arr.sort(key=lambda x: x[1], reverse=True)
        if len(arr) < 2:
            return 0, 0, 0, 0
        deltapeak1, h1 = arr[0]
        deltapeak2, h2 = arr[1]
        rmsA = np.std(crossCorr - np.correlate(template_flux_proc, template_flux_proc, mode='full')[::-1][int(nw / 2) - int(deltapeak1 - nw / 2):int(nw + nw / 2) - int(deltapeak1 - nw / 2)] / max(np.correlate(template_flux_proc, template_flux_proc, mode='full')))
        r = abs((h1 - rmsA) / (np.sqrt(2) * rmsA))
        fom = (h1 - 0.05) ** 0.75 * (h1 / h2) if h2 != 0 else 0
        shift = int(deltapeak1 - nw / 2)
        iminindex, imaxindex = self._min_max_index(input_flux_proc, nw)
        tminindex, tmaxindex = self._min_max_index(template_flux_proc, nw)
        overlapminindex = int(max(iminindex + shift, tminindex))
        overlapmaxindex = int(min(imaxindex - 1 + shift, tmaxindex - 1))
        minWaveOverlap = wave[overlapminindex]
        maxWaveOverlap = wave[overlapmaxindex]
        lap = np.log(maxWaveOverlap / minWaveOverlap) if minWaveOverlap > 0 else 0
        rlap = 5 * r * lap
        fom = fom * lap
        return r, lap, rlap, fom

    def _min_max_index(self, flux: np.ndarray, nw: int) -> tuple:
        minindex, maxindex = (0, nw - 1)
        zeros = np.where(flux == 0)[0]
        j = 0
        for i in zeros:
            if (i != j):
                break
            j += 1
            minindex = j
        j = int(nw) - 1
        for i in zeros[::-1]:
            if (i != j):
                break
            j -= 1
            maxindex = j
        return minindex, maxindex
