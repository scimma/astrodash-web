import numpy as np
from scipy.signal import argrelmax
from scipy.fft import fft
from .astrodash_backend import mean_zero_spectra, get_training_parameters
from .redshift_estimator import get_median_redshift
from .utils import get_nonzero_minmax, get_redshift_axis
import logging

logger = logging.getLogger("rlap_calculator")

def shift_to_rest_frame(wave, flux, redshift):
    """Shift observed spectrum to rest-frame using the given redshift."""
    rest_wave = wave / (1 + redshift)
    return rest_wave, flux

class RlapCalculator:
    def __init__(self, inputFlux, templateFluxes, templateNames, wave, inputMinMaxIndex, templateMinMaxIndexes):
        self.templateFluxes = templateFluxes
        self.templateNames = templateNames
        self.wave = wave
        pars = get_training_parameters()
        w0, w1, self.nw = pars['w0'], pars['w1'], pars['nw']
        self.inputFlux = mean_zero_spectra(inputFlux, inputMinMaxIndex[0], inputMinMaxIndex[1], self.nw)
        self.templateMinMaxIndexes = templateMinMaxIndexes
        self.dwlog = np.log(w1 / w0) / self.nw

    def _cross_correlation(self, templateFlux, templateMinMaxIndex):
        templateFlux = mean_zero_spectra(templateFlux, templateMinMaxIndex[0], templateMinMaxIndex[1], self.nw)
        inputfourier = fft(self.inputFlux)
        tempfourier = fft(templateFlux)
        product = inputfourier * np.conj(tempfourier)
        xCorr = fft(product)
        rmsInput = np.std(inputfourier)
        rmsTemp = np.std(tempfourier)
        xCorrNorm = (1. / (self.nw * rmsInput * rmsTemp)) * xCorr
        rmsXCorr = np.std(product)
        xCorrNormRearranged = np.concatenate((xCorrNorm[int(len(xCorrNorm) / 2):], xCorrNorm[0:int(len(xCorrNorm) / 2)]))
        crossCorr = np.correlate(self.inputFlux, templateFlux, mode='full')[::-1][int(self.nw / 2):int(self.nw + self.nw / 2)] / max(np.correlate(self.inputFlux, templateFlux, mode='full'))
        try:
            deltapeak, h = self._get_peaks(crossCorr)[0]
            shift = int(deltapeak - self.nw / 2)
            autoCorr = np.correlate(templateFlux, templateFlux, mode='full')[::-1][int(self.nw / 2) - shift:int(self.nw + self.nw / 2) - shift] / max(np.correlate(templateFlux, templateFlux, mode='full'))
            aRandomFunction = crossCorr - autoCorr
            rmsA = np.std(aRandomFunction)
        except IndexError as err:
            logger.error(f"Error: Cross-correlation is zero, probably caused by empty spectrum. {err}")
            rmsA = 1
        return xCorr, rmsInput, rmsTemp, xCorrNorm, rmsXCorr, xCorrNormRearranged, rmsA

    def _get_peaks(self, crosscorr):
        peakindexes = argrelmax(crosscorr)[0]
        ypeaks = [abs(crosscorr[i]) for i in peakindexes]
        arr = list(zip(peakindexes, ypeaks))
        arr.sort(key=lambda x: x[1], reverse=True)
        return arr

    def _calculate_r(self, crosscorr, rmsA):
        peaks = self._get_peaks(crosscorr)
        if len(peaks) < 2:
            return 0, 0, 0
        deltapeak1, h1 = peaks[0]
        deltapeak2, h2 = peaks[1]
        r = abs((h1 - rmsA) / (np.sqrt(2) * rmsA))
        fom = (h1 - 0.05) ** 0.75 * (h1 / h2) if h2 != 0 else 0
        return r, deltapeak1, fom

    def calculate_rlap(self, crosscorr, rmsAntisymmetric, templateFlux):
        r, deltapeak, fom = self._calculate_r(crosscorr, rmsAntisymmetric)
        shift = int(deltapeak - self.nw / 2)
        iminindex, imaxindex = self.min_max_index(self.inputFlux)
        tminindex, tmaxindex = self.min_max_index(templateFlux)
        overlapminindex = int(max(iminindex + shift, tminindex))
        overlapmaxindex = int(min(imaxindex - 1 + shift, tmaxindex - 1))
        minWaveOverlap = self.wave[overlapminindex]
        maxWaveOverlap = self.wave[overlapmaxindex]
        lap = np.log(maxWaveOverlap / minWaveOverlap) if minWaveOverlap > 0 else 0
        rlap = 5 * r * lap
        fom = fom * lap
        return r, lap, rlap, fom

    def min_max_index(self, flux):
        minindex, maxindex = (0, self.nw - 1)
        zeros = np.where(flux == 0)[0]
        j = 0
        for i in zeros:
            if (i != j):
                break
            j += 1
            minindex = j
        j = int(self.nw) - 1
        for i in zeros[::-1]:
            if (i != j):
                break
            j -= 1
            maxindex = j
        return minindex, maxindex

    def rlap_score(self, tempIndex):
        xcorr, rmsinput, rmstemp, xcorrnorm, rmsxcorr, xcorrnormRearranged, rmsA = self._cross_correlation(
            self.templateFluxes[tempIndex].astype('float'), self.templateMinMaxIndexes[tempIndex])
        crosscorr = xcorrnormRearranged
        r, lap, rlap, fom = self.calculate_rlap(crosscorr, rmsA, self.templateFluxes[tempIndex])
        return r, lap, rlap, fom

    def rlap_label(self):
        if not np.any(self.inputFlux):
            return "No flux", True
        self.zAxis = get_redshift_axis(self.nw, self.dwlog)
        rlapList = []
        for i in range(len(self.templateNames)):
            r, lap, rlap, fom = self.rlap_score(tempIndex=i)
            rlapList.append(rlap)
        rlapMean = round(np.mean(rlapList), 2)
        rlapLabel = str(rlapMean)
        rlapWarning = rlapMean < 6
        return rlapLabel, rlapWarning

def calculate_rlap_with_redshift(wave, flux, templateFluxes, templateNames, templateMinMaxIndexes, inputMinMaxIndex, redshift=None):
    """
    Calculate RLap score for the input spectrum using the best-match template.
    If redshift is not provided, estimate it using the templates.
    Returns (rlap_score, used_redshift, rlap_warning)
    """
    pars = get_training_parameters()
    w0, w1, nw = pars['w0'], pars['w1'], pars['nw']
    dwlog = np.log(w1 / w0) / nw

    # If redshift is not provided, estimate it using the templates
    if redshift is None:
        logger.info("Estimating redshift for RLap calculation using best-match template(s).")
        est_redshift, _, _, _ = get_median_redshift(
            flux, templateFluxes, nw, dwlog, inputMinMaxIndex, templateMinMaxIndexes, templateNames
        )
        if est_redshift is None:
            logger.error("Redshift estimation failed. RLap will be calculated in observed frame.")
            est_redshift = 0.0
        redshift = est_redshift
    else:
        logger.info(f"Using provided redshift {redshift} for RLap calculation.")

    # Shift input spectrum to rest-frame
    rest_wave = wave / (1 + redshift)
    # Interpolate flux onto the log-wavelength grid if needed (assume input is already log-binned)
    # For faithful port, we assume input is already on the correct grid

    # Calculate RLap
    rlap_calc = RlapCalculator(flux, templateFluxes, templateNames, rest_wave, inputMinMaxIndex, templateMinMaxIndexes)
    rlap_label, rlap_warning = rlap_calc.rlap_label()
    return rlap_label, redshift, rlap_warning

def compute_rlap_for_matches(matches, best_match, log_wave, input_flux_log, template_fluxes, template_names, template_minmax_indexes, known_z):
    """
    Compute RLap for the best match and attach to matches and best_match dicts.
    """
    # Find the best match index
    if not matches:
        return matches, best_match
    best_match_idx = np.argmax([m['probability'] for m in matches])
    best = matches[best_match_idx]
    sn_type = best['type']
    age = best['age']
    # Find the correct template(s) for this type/age
    # (Assume template_fluxes, template_names, template_minmax_indexes are already filtered for the best match)
    input_minmax_index = get_nonzero_minmax(input_flux_log)
    rlap_label, used_redshift, rlap_warning = calculate_rlap_with_redshift(
        log_wave, input_flux_log, template_fluxes, template_names, template_minmax_indexes, input_minmax_index,
        redshift=best['redshift'] if known_z else None
    )
    # Attach RLap to all matches (or just best match)
    for m in matches:
        m['rlap'] = rlap_label
        m['rlap_warning'] = rlap_warning
    best_match['rlap'] = rlap_label
    best_match['rlap_warning'] = rlap_warning
    return matches, best_match
