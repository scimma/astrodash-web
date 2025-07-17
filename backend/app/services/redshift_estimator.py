import numpy as np
from scipy.signal import argrelmax
import logging

logger = logging.getLogger("redshift_estimator")


def mean_zero_spectra(flux, min_idx, max_idx, nw):
    out = np.zeros(nw)
    region = flux[min_idx:max_idx+1]
    mean = np.mean(region) if len(region) > 0 else 0
    out[min_idx:max_idx+1] = region - mean
    return out

def calc_redshift_from_crosscorr(crossCorr, nw, dwlog):
    # Find max peak while ignoring peaks that lead to negative redshifts
    deltaPeak = np.argmax(crossCorr[:int(nw // 2) + 1])
    zAxisIndex = np.concatenate((np.arange(-nw // 2, 0), np.arange(0, nw // 2)))
    if deltaPeak <= nw // 2:
        z = (np.exp(np.abs(zAxisIndex) * dwlog) - 1)[deltaPeak]
    else:
        z = -(np.exp(np.abs(zAxisIndex) * dwlog) - 1)[deltaPeak]
    return z, crossCorr

def cross_correlation(inputFlux, tempFlux, nw, tempMinMaxIndex):
    # Both inputFlux and tempFlux should be mean-zeroed and apodized
    inputfourier = np.fft.fft(inputFlux)
    tempfourier = np.fft.fft(tempFlux)
    product = inputfourier * np.conj(tempfourier)
    xCorr = np.fft.fft(product)
    rmsInput = np.std(inputfourier)
    rmsTemp = np.std(tempfourier)
    xCorrNorm = (1. / (nw * rmsInput * rmsTemp)) * xCorr
    xCorrNormRearranged = np.concatenate((xCorrNorm[int(len(xCorrNorm) / 2):], xCorrNorm[0:int(len(xCorrNorm) / 2)]))
    crossCorr = np.correlate(inputFlux, tempFlux, mode='full')[::-1][int(nw / 2):int(nw + nw / 2)] / max(np.correlate(inputFlux, tempFlux, mode='full'))
    return crossCorr

def get_redshift(inputFlux, tempFlux, nw, dwlog, tempMinMaxIndex):
    crossCorr = cross_correlation(inputFlux, tempFlux, nw, tempMinMaxIndex)
    redshift, crossCorr = calc_redshift_from_crosscorr(crossCorr, nw, dwlog)
    return redshift, crossCorr

def get_median_redshift(inputFlux, tempFluxes, nw, dwlog, inputMinMaxIndex, tempMinMaxIndexes, tempNames, outerVal=0.5):
    inputFlux = mean_zero_spectra(inputFlux, inputMinMaxIndex[0], inputMinMaxIndex[1], nw)
    redshifts = []
    crossCorrs = {}
    for i, tempFlux in enumerate(tempFluxes):
        assert tempFlux[0] == outerVal or tempFlux[0] == 0.0
        redshift, crossCorr = get_redshift(inputFlux, tempFlux - outerVal, nw, dwlog, tempMinMaxIndexes[i])
        redshifts.append(redshift)
        crossCorrs[tempNames[i]] = crossCorr
    if redshifts:
        medianIndex = np.argsort(redshifts)[len(redshifts) // 2]
        medianRedshift = redshifts[medianIndex]
        medianName = tempNames[medianIndex]
        try:
            stdRedshift = np.std(redshifts)
        except Exception as e:
            logger.error(f"Error calculating redshift error: {e}")
            stdRedshift = None
    else:
        return None, None, None, None
    if len(redshifts) >= 10:
        redshiftError = np.std(redshifts)
    else:
        redshiftError = None
    return medianRedshift, crossCorrs, medianName, stdRedshift
