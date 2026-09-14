"""Acoustics helpers (A-weighting / LAeq) with no amods equivalent."""
import numpy as np
from scipy.signal import bilinear, lfilter


def a_weighting(sample_rate):
    """
    Design a digital A-weighting filter (approximate).
    Returns (b, a) filter coefficients for use with scipy.signal.lfilter.
    """
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    a1000 = 1.9997

    # Analog coefficients
    nums = [(2 * np.pi * f4) ** 2 * (10 ** (a1000 / 20)), 0, 0, 0, 0]
    dens = np.polymul([1 + 2 * np.pi * f4, (2 * np.pi * f4) ** 2],
                       np.polymul([1 + 2 * np.pi * f1, 2 * np.pi * f1],
                                  np.polymul([1 + 2 * np.pi * f3, 2 * np.pi * f3],
                                             [1 + 2 * np.pi * f2, 2 * np.pi * f2])))

    b, a = bilinear(nums, dens, sample_rate)
    return b, a


def compute_LAeq(signal, sample_rate):
    """Compute the A-weighted equivalent continuous sound level (dBA) of ``signal``."""
    if signal.ndim > 1:
        signal = signal.mean(axis=0)
    b, a = a_weighting(sample_rate)
    signal_a = lfilter(b, a, signal)
    rms = np.sqrt(np.mean(signal_a ** 2))
    laeq = 20 * np.log10(rms + 1e-12)
    return laeq
