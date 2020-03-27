"""
Created: March 2020
Latest update:  March 2020
@author: Praveer Nidamaluri
Library for generic time history analysis functions.
"""

# %% Import Necessary Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from numpy.fft import rfft, rfftfreq, irfft

# %% Define global properties

# ## Time History Analysis Utility Functions


def low_pass_filter(signal, lp_frq, dt=0.005, time=None, zero_pad=True):
    """
    Function to low-pass filter a signal in the frequency domain,
    by setting all fourier terms larger than `lp_freq` to 0.

    :param signal: list-like object with the input signal.
    :param lp_frq: Low-pass cut-off frequency.
    :param dt: float object with the timestep. Defaults to 0.005s.
                Only used if `time` is `None`.
    :param time: list-like object with the input signal. Default to None.
    :param zero_pad: Boolean. If `True`, the signal is padded with zeros of
                    length `2*len(signal)`. This should be used if the input
                    is non-repetitive. IF `False`, the input is not zero-padded.
                    Hence, the fft assumes that the signal repeats.
    :return: lp_signal : 1D numpy array with the low-pass filtered signal.
    """

    if time is not None:
        dt = time[1]-time[0]
    if zero_pad:
        n = 3*len(signal)
    else:
        n = len(signal)
    frq = rfftfreq(n, d=dt)
    signal_fft = rfft(signal, n=n)
    lp_fft = signal_fft*(frq <= lp_frq)
    lp_signal = irfft(lp_fft, n)[:len(signal)]
    return lp_signal
