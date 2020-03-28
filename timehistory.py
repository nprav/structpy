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


def animate_plot_2d(x, y, num_frames=100, anim_length=5, figsize=None, title=None,
                    xlabel='x', ylabel='y', l_kwargs=None, m_kwargs=None):
    """
    Simple Matplotlib wrapper function that generates a 2d scatter plot animation with
    1d x and y data.

    :param x: 1d list-like x-axis data.
    :param y: 1d list-like y-axis data.
    :param num_frames: int - number of frames in the animation.
    :param anim_length: float - the length of the animation in seconds.
    :param figsize: (int, int) - tuple with typical matplotlib figsize
                    specification
    :param title: string - title of the plot. Defaults to None.
    :param xlabel: string - x-axis label of the plot. Defaults to 'x'.
    :param ylabel: string - y-axis label of the plot. Defaults to 'y'.
    :param l_kwargs: dict - Matplotlib Line2D properties to be used as
                    keyword arguments for the main line plot.
    :param m_kwargs: dict - Matplotlib Line2D properties to be used as
                    keyword arguments for the single marker at end of
                    the plot in each frame.
    :return: IPython.display HTML object with a video of the animation.
    """

    # Default arguments:
    if l_kwargs is None:
        l_kwargs = {}
    if m_kwargs is None:
        m_kwargs = {}

    # Set up the figure and axes
    if figsize is None:
        fig1 = plt.figure()
    else:
        fig1 = plt.figure(figsize=figsize)
    axis = fig1.gca()
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if title is not None:
        axis.set_title(title)

    # Set up the matplotlib artists
    l, = axis.plot([], [], **l_kwargs)
    m, = axis.plot([], [], 'o', **m_kwargs)
    axis.set_xlim((x.min(), x.max()))
    axis.set_ylim((y.min(), y.max()))

    # Define the animation parameters
    step = len(x)//(num_frames-1)
    frames = list(range(0, len(x), step))
    if not frames[-1] == len(x)-1:
        frames.append(len(x)-1)
    interval = anim_length/num_frames*1000

    # Define artist update function
    def update_line(frame, line, marker):
        line.set_data(x[:frame], y[:frame])
        marker.set_data(x[frame], y[frame])
        return line, marker

    # Generate the html animation
    line_ani = animation.FuncAnimation(fig1, update_line, frames,
                                       fargs=(l, m), interval=interval, blit=True)
    return HTML(line_ani.to_jshtml())
