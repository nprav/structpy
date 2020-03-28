"""
Created: March 2020
Latest update:  March 2020
@author: Praveer Nidamaluri
Library for generic time history analysis functions.
"""

# %% Import Necessary Modules
import numpy as np
import pandas as pd
from scipy.integrate import cumtrapz
from numpy.fft import rfft, rfftfreq, irfft
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# %% Define global properties

# ## Time History Analysis Utility Functions


def get_rfft(signal, dt=0.005, time=None, zero_pad=True):
    """
    Function to return the real absolute fft and frequencies of
    a given signal.

    :param signal: 1d list-like object with the input signal.
    :param dt: float object with the timestep. Defaults to 0.005s.
                Only used if `time` is `None`.
    :param time: 1d list-like object with the input signal. Default to None.
    :param zero_pad: Boolean. If `True`, the signal is padded with zeros of
                    length `2*len(signal)`. This should be used if the input
                    is non-repetitive. IF `False`, the input is not zero-padded.
                    Hence, the fft assumes that the signal repeats.
    :return:
        - frq - 1d ndarray with frequencies.
        - signal_fft - 1d ndarray with absolute fft values.
    """

    if time is not None:
        dt = time[1]-time[0]
    if zero_pad:
        n = 3*len(signal)
    else:
        n = len(signal)
    frq = rfftfreq(n, d=dt)
    signal_fft = np.abs(rfft(signal, n=n))
    return frq, signal_fft


def low_pass_filter(signal, lp_frq, dt=0.005, time=None, zero_pad=True):
    """
    Function to low-pass filter a signal in the frequency domain,
    by setting all fourier terms larger than `lp_freq` to 0.

    :param signal: 1d list-like object with the input signal.
    :param lp_frq: Low-pass cut-off frequency.
    :param dt: float object with the timestep. Defaults to 0.005s.
                Only used if `time` is `None`.
    :param time: 1d list-like object with the input signal. Default to None.
    :param zero_pad: Boolean. If `True`, the signal is padded with zeros of
                    length `2*len(signal)`. This should be used if the input
                    is non-repetitive. IF `False`, the input is not zero-padded.
                    Hence, the fft assumes that the signal repeats.
    :return: 1D ndarray with the low-pass filtered signal.
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


def animate_plot_2d(x, y):
    def update_line(frame, line):
        line.set_data(x[:frame], y[:frame])
        return line,
    fig1 = plt.figure()
    l, = plt.plot([], [])
    plt.xlim((x.min(), x.max()))
    plt.ylim((y.min(), y.max()))
    line_ani = animation.FuncAnimation(fig1, update_line, min(len(x), 100),
                                       fargs=(l,), interval=50, blit=True)
    return HTML(line_ani.to_jshtml())