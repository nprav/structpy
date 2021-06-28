'''
Created: Nov 2017
Latest update:  June 2020
@author: Praveer Nidamaluri

Module for generating and broadbanding acceleration response spectra.
'''

# %% Import Necessary Modules
import numpy as np
from time import perf_counter
from itertools import accumulate


# %% Response Spectrum Generation

def get_step_matrix(w, zeta, dt):
    """
    Calculate the A, B matrices from [1] based on the input
    angular frequency `w`, critical damping ratio, `zeta`,
    and timestep, `dt`.

    For use with the `step_resp_spect` function that generates
    response spectra by the step-by-step method.

    Parameters
    ----------
    w : float
        Angular frequency in rads/s.

    zeta : float
        Critical damping ratio (dimensionless).

    dt : float
        Timestep in s.

    Returns
    -------
    A : (2,2) ndarray
        Array to be matrix-multiplied by the [x_i, xdot_i] vector.

    B : (2,2) ndarray
        Array to be matrix-multiplied by the [a_i, a_(i+1)] vector.

    References
    ----------
    .. [1] Nigam, Jennings, April 1969. Calculation of response Sepctra
        from Stong-Motion Earthquake Records. Bulletin of the Seismological
        Society of America. Vol 59, no. 2.
    """

    A = np.zeros((2, 2))
    B = np.zeros((2, 2))

    exp = np.exp(-zeta*w*dt)
    zsqt = (1-zeta**2)**0.5
    sin = np.sin(w*zsqt*dt)
    cos = np.cos(w*zsqt*dt)

    A[0, 0] = exp*(cos+sin*zeta/zsqt)
    A[0, 1] = exp/(w*zsqt)*sin
    A[1, 0] = -w/zsqt*exp*sin
    A[1, 1] = exp*(cos-sin*zeta/zsqt)

    t1 = (2*zeta**2 - 1)/w**2/dt
    t2 = 2*zeta/w**3/dt

    B[0, 0] = exp*(sin/(w*zsqt)*(t1 + zeta/w) + cos*(t2 + 1/w**2)) - t2
    B[0, 1] = -exp*(sin/(w*zsqt)*t1 + cos*t2) - 1/w**2 + t2
    B[1, 0] = exp*((t1 + zeta/w)*(cos - sin*zeta/zsqt)
             - (t2 + 1/w**2)*(sin*w*zsqt + cos*zeta*w)) + 1/w**2/dt
    B[1, 1] = -exp*(t1*(cos - sin*zeta/zsqt)
             - t2*(sin*w*zsqt + cos*zeta*w)) - 1/w**2/dt

    return A, B


def step_resp_spect(acc, time_a, zeta=0.05, ext=True, verbose=False):
    '''
    Generate acceleration response spectrum by the step-by-step method [1].
    The algorithm is programmed to match that from SHAKE2000. The theory behind
    the algorithm assumes a 'segmentally-linear' acceleration time history (TH).
    Hence, the implicit assumption is that the time history nyquist frequency is
    much higher than the highest frequency within the TH.

    Use the `fft_resp_spect` method if there is frequency content close to the
    nyquist frequency. Or, use `scipy.signal.resample` to up-sample the acc. TH
    prior to using `step_resp_spect`.

    Output frequencies are loglinearly spaced as follows:
        - [0.1Hz, 1Hz] : 12 points
        - [1Hz, 10Hz] : 50 points
        - [10Hz, 100Hz] : 25 points
        - [100Hz, 1000Hz] : 15 points (only if `ext` is True)

    Parameters
    ----------
    acc : 1D array_like
        Input acceleration time history (assumed to be in g's).

    time_a : 1D array_like
        Input time values for the acceleration time history, `acc`.

    zeta : float, optional
        Critical damping ratio (dimensionless). Defaults to 0.05.

    ext : bool, optional
        Defines whether the RS is calculated to 100Hz or 1000Hz. Defaults
        to True.
            - ext = True : Frequency range is [0.1Hz, 1000Hz]
            - ext = False : Frequency range is [0.1Hz - 100Hz]

    verbose: bool, optional
            If True, the function prints the total time taken, and the time
            per frequency.

    Returns
    -------
    rs : 1D ndarray
        Array with spectral accelerations (same units as input acc).

    frq : 1D ndarray
        Array with frequencies in Hz.

    References
    ----------
    .. [1] Nigam, Jennings, April 1969. Calculation of response Spectra
        from Strong-Motion Earthquake Records. Bulletin of the Seismological
        Society of America. Vol 59, no. 2.
    '''

    t0 = perf_counter()

    # Set up list of frequencies on which to calculate response spectra:
    frq = np.logspace(-1, 0, num=12, endpoint=False)
    frq = np.append(frq, np.logspace(0, 1, num=50, endpoint=False))
    if ext:
        frq = np.append(frq, np.logspace(1, 2, num=25, endpoint=False))
        frq = np.append(frq, np.logspace(2, 3, num=15, endpoint=True))
    else:
        frq = np.append(frq, np.logspace(1, 2, num=25, endpoint=True))

    # Instantiate angular frequency and spectral acceleration arrays
    w = frq*2*np.pi
    rs = 0*w

    # Define timestep from input signal
    dt = time_a[1] - time_a[0]

    # Define utility function to be used with itertools.accumulate
    def func(x, a):
        return np.dot(A, x) + np.dot(B, a)

    # Calculate response for a spring with each wn
    for k, wn in enumerate(w):

        # Calculate response acceleration time history
        A, B = get_step_matrix(wn, zeta, dt)
        act = np.column_stack((acc[:-1], acc[1:]))
        act = np.append(np.array([[0, 0], [0, acc[0]]]),
                        act,
                        axis=0)
        x = np.array(list(accumulate(act, func)))
        temp = -np.array([wn**2, 2*zeta*wn])
        z = np.dot(x, temp)
        rs[k] = np.max(np.absolute(z))

    t1 = perf_counter()
    t_net = t1 - t0

    if verbose:
        print("RS done. Time taken = {:.5f}s".format(t_net),
              "\ntime per iteration = {:.5f}s".format(t_net/len(w)))

    return rs, frq


def fft_resp_spect(acc, time_a, zeta=0.05, ext=True, verbose=False):
    '''
    Generate acceleration response spectrum using a frequency domain
    method. This is physically accurate if the true acceleration time
    history has no frequency content higher than the nyquist frequency
    of the input acceleration.

    Output frequencies are loglinearly spaced as follows:
        - [0.1Hz, 1Hz] : 30 points
        - [1Hz, 10Hz] : 50 points
        - [10Hz, 100Hz] : 50 points
        - [100Hz, 1000Hz] : 30 points (only if `ext` is True)

    Parameters
    ----------
    acc : 1D array_like
        Input acceleration time history (assumed to be in g's).

    time_a : 1D array_like
        Input time values for the acceleration time history, `acc`.

    zeta : float, optional
        Critical damping ratio (dimensionless). Defaults to 0.05.

    ext : bool, optional
        Defines whether the RS is calculated to 100Hz or 1000Hz. Defaults
        to True.
            - ext = True : Frequency range is [0.1Hz, 1000Hz]
            - ext = False : Frequency range is [0.1Hz - 100Hz]

    verbose: bool, optional
            If True, the function prints the total time taken, and the time
            per frequency.

    Returns
    -------
    rs : 1D ndarray
        Array with spectral accelerations (same units as input acc).

    frq : 1D ndarray
        Array with frequencies in Hz.
    '''

    t0 = perf_counter()

    # Set up list of frequencies on which to calculate response spectra:
    frq = np.logspace(-1, 0, num=30, endpoint=False)
    frq = np.append(frq, np.logspace(0, 1, num=50, endpoint=False))
    if ext:
        frq = np.append(frq, np.logspace(1, 2, num=50, endpoint=False))
        frq = np.append(frq, np.logspace(2, 3, num=30, endpoint=True))
    else:
        frq = np.append(frq, np.logspace(1, 2, num=50, endpoint=True))

    # Instantiate angular frequency and spectral acceleration arrays
    w = frq*2*np.pi
    rs = 0*w

    # Define minimum timestep from input signal
    dt_min = time_a[1] - time_a[0]

    # Calculate n, the integer to determine 0 padding at the end
    # of the time history; making n a power of 2 improves the
    # efficiency of the fft algorithm
    n = len(acc)
    n_fft = int(2 ** (np.ceil(np.log(1.5 * n) / np.log(2))))

    # Get n for upsampling by sinc-interpolating so there are
    # `multiplier` times as many points
    multiplier = 10

    # Get FFT of input acceleration
    xgfft = np.fft.rfft(acc, n_fft)
    frqt = np.fft.rfftfreq(n_fft, d=dt_min)

    # Calculate response for a spring with each wn
    for k, wn in enumerate(w):

        # Angular frequencies of fft
        wf = frqt*2*np.pi

        # Displacement of spring mass (fourier terms)
        xfft = -xgfft/(-wf**2 + 2*zeta*wn*1j*wf + wn**2)

        # Relative acceleration of spring mass (fourier terms)
        accfft = -xfft*wf**2

        # Absolute acceleration of spring mass (fourier terms)
        abs_accfft = accfft + xgfft

        # Get absolute acceleration of spring mass (time domain)
        # Up-sample so that the final time history is sinc-
        # interpolated with `n_multiplier` total points
        a = np.fft.irfft(abs_accfft, n=multiplier*n_fft) * multiplier

        # Peak absolute acceleration of spring mass
        rs[k] = np.max(np.absolute(a))

    t1 = perf_counter()
    t_net = t1 - t0

    if verbose:
        print("RS done. Time taken = {:.5f}s".format(t_net),
              "\ntime per iteration = {:.5f}s".format(t_net/len(w)))

    return rs, frq
