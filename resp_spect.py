'''
Created: Nov 2017
Latest update:  Aug 2019
@author: Praveer Nidamaluri

Module for generating and broadbanding acceleration response spectra.
'''

# %% Import Necessary Modules
import numpy as np
import time as timer
from itertools import accumulate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize


# %% Response Spectrum Generation

def get_step_matrix(w, zeta=0.05, dt=0.005):
    '''
    For use with the step_resp_spect() function that generates
    response spectra in the step-by-step method.
    '''

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
    B[1, 0] = exp*((t1 + zeta/w)*(cos - sin*zeta/zsqt) - (t2 + 1/w**2)*(sin*w*zsqt + cos*zeta*w)) + 1/w**2/dt
    B[1, 1] = -exp*(t1*(cos - sin*zeta/zsqt) - t2*(sin*w*zsqt + cos*zeta*w)) - 1/w**2/dt

    return A, B


def step_resp_spect(acc,time_a,zeta=0.05,ext=True,plot=False,max_nyq=500,accuracy = 0.2):
    '''
    Only accurate up till 200Hz or so - for more accuracy must manually change the interpolation step/nyquist
    freq.
    Input: acceleration array, time array of same length of acc., critical damping ratio in
    decimal, ext = boolean (if true, frequencies between 100Hz and 1000Hz are also included), plot = present
    or not, accuracy = between 0, and 1 (1 = most accurate, but slow, 0 = less accurate but fast)
    Output: array with response in g's, array with frequencies (in Hz)
    Also prints time taken, and displays output spectrum

    Adaptation of method described in "Calculation of response Sepctra from mMStong-Motion Earthquake Records"
    Nigam, Jennings, Bulletin of the Seismological Society of America vol 59, no. 2, April 1969.

    '''

    t0 = timer.clock()

    # Set up speed vs accuracy variable:
    try:
        accuracy = max(min(1,accuracy),0)
        frq_mult = 5 + 10*accuracy
    except:
        frq_mult = 7

    # Set up list of frequencies on which to calculate response spectra:
    frq = np.logspace(-1,0,num=12,endpoint=False)
    frq = np.append(frq,np.logspace(0,1,num=50,endpoint=False))
    if ext:
        frq = np.append(frq,np.logspace(1,2,num=25,endpoint=False))
        frq = np.append(frq,np.logspace(2,3,num = 15,endpoint=True))
    else:
        frq = np.append(frq,np.logspace(1,2,num=25,endpoint=True))

    w = frq*2*np.pi
    rs = 0*w

    dt_max = 1/(2*max_nyq)

    func = lambda x,a : np.dot(A,x) + np.dot(B,a)

    # Calculate response for a spring with each wn
    for k,wn in enumerate(w):
        # Interpolate time/acceleration vector to reduce total number of calculations
        nyq = max(frq_mult*frq[k],20)
        dt = max(1/(2*nyq),dt_max)
        dt_tm = np.arange(0,time_a[-1],dt)
        dt_acc = np.interp(dt_tm,time_a,acc)

        # Calculate response acceleration time history
        A,B = get_step_matrix(wn,zeta,dt)
        act = np.column_stack((dt_acc[:-1],dt_acc[1:]))
        act = np.append(np.array([[0,0],[0,dt_acc[0]]]),act,axis=0)
        x = np.array(list(accumulate(act,func)))
        temp = -np.array([wn**2,2*zeta*wn])
        z = np.dot(x,temp)
        rs[k] = np.max(np.absolute(z))

    t1 = timer.clock()
    t_net = t1 - t0

    print("RS done. Time taken = ", t_net, "\ntime per iteration = ", t_net/len(w))

    if plot:
        plt.semilogx(frq,rs,'.-b')
        plt.grid(b=True,which = 'both', axis = 'both', alpha = 0.5)
        plt.title("Response Spectrum")
        plt.xlabel("Frequency/Hz")
        plt.ylabel("Acceleration/g's")
        k = "%s, %s" %(round(frq[-1]),round(rs[-1],2))
        plt.annotate(k,xy=(frq[-1],rs[-1]),)
        plt.show()

    return rs, frq


def fft_resp_spect(acc,time_a,zeta=0.05,ext=True,plot=False,max_nyq=500,accuracy = 0.2):
    '''
    Only accurate up till 200Hz or so - for more accuracy must manually change the interpolation step/nyquist
    freq.
    Input: acceleration array, time array of same length of acc., critical damping ratio in
    decimal, ext = boolean (if true, frequencies between 100Hz and 1000Hz are also included), plot = present
    or not, accuracy = between 0, and 1 (1 = most accurate, but slow, 0 = less accurate but faster)
    Output: array with response in g's, array with frequencies (in Hz)
    Also prints time taken, and displays output spectrum

    '''

    t0 = timer.clock()

    # Set up speed vs accuracy variable:
    try:
        accuracy = max(min(1,accuracy),0)
        frq_mult = 5 + 10*accuracy
    except:
        frq_mult = 7

    # Set up list of frequencies on which to calculate response spectra:
    frq = np.logspace(-1,0,num=12,endpoint=False)
    frq = np.append(frq,np.logspace(0,1,num=50,endpoint=False))
    if ext:
        frq = np.append(frq,np.logspace(1,2,num=25,endpoint=False))
        frq = np.append(frq,np.logspace(2,3,num = 15,endpoint=True))
    else:
        frq = np.append(frq,np.logspace(1,2,num=25,endpoint=True))

    w = frq*2*np.pi
    rs = 0*w

    dt_max = 1/(2*max_nyq)

    # Calculate response for a spring with each wn
    for k,wn in enumerate(w):
        # Interpolate time/acceleration vector to reduce total number of calculations
        nyq = max(frq_mult*frq[k],20)
        dt = max(1/(2*nyq),dt_max)
        dt_tm = np.arange(0,time_a[-1],dt)
        dt_acc = np.interp(dt_tm,time_a,acc)

        # Calculate n, the integer to determine 0 padding at the end of the time history;
        # making n a power of 2 improves the efficiency of the fft algorithm
        n = int(2**(np.ceil(np.log(1.5*len(dt_acc))/np.log(2))))

        # Solve for acceleration response
        xgfft = np.fft.rfft(dt_acc,n)
        frqt = np.fft.rfftfreq(n, d = dt_tm[-1]/len(dt_tm))
        xfft = 0*xgfft
        accfft = 0*xgfft

        for i,f in enumerate(frqt):
            wf = f*2*np.pi
            xfft[i] = -xgfft[i]/(-wf**2 + 2*zeta*wn*1j*wf + wn**2)
            accfft[i] = -xfft[i]*wf**2

        a = np.fft.irfft(accfft)
        abs_a = a[:len(dt_tm)] + dt_acc
        rs[k] = np.max(np.absolute(abs_a))

    t1 = timer.clock()
    t_net = t1 - t0

    print("RS done. Time taken = ", t_net, "\ntime per iteration = ", t_net/len(w))

    if plot:
        plt.semilogx(frq,rs,'.-b')
        plt.grid(b=True,which = 'both', axis = 'both', alpha = 0.5)
        plt.title("Response Spectrum")
        plt.xlabel("Frequency/Hz")
        plt.ylabel("Acceleration/g's")
        k = "%s, %s" %(round(frq[-1]),round(rs[-1],2))
        plt.annotate(k,xy=(frq[-1],rs[-1]),)
        plt.show()

    return rs, frq


# %% Response Spectrum Post-processing - Broadbanding functions


def simple_broadband(frq, rs, npts=8, ratio=0.15, window=8, amp=1.02):
    """Generate a simplified and broadbanded response spectrum
    from a given response spectrum (RS).

    Simple algorithm that attempts to curve-fit a piecewise loglinear
    function with N=npts points to a broadbanded version of the inputs RS.

    This function has been superceded by 'optimal_broadband' and
    'iter_optimal_broadband' which are more robust algorithms
    with more features.

    Parameters
    ----------

    frq : 1D list/tuple/ndarray
        Frequencies of input response spectrum to be broadbanded.

    rs: 1D list/tuple/ndarray
        Spectral acceleration values of input response spectrum
        to be broadbanded. Object should have the same length as frq.

    npts : int, optional
        Positive integer >= 2 that specifies the number of points the
        simplified response spectrum should have. Defaults to 8.

    ratio : float, optional
        Parameter that defines the broadbanding. The input response spectrum
        is shifted by +- ratio. The frequencies in the input RS are multiplied
        by (1+ratio) and (1-ratio). The final simplified, broad-banded
        RS is generated to bound the inputs RS and shifted input RSs.
        The ratio parameter therefore determines how wide the final RS peaks
        should be. Defaults to 0.15 per ASCE 4-98 or Reg. Guide 1.122.

    window : int, optional
        Parameter that defines the window of rolling_max and rolling_mean used
        in the broadband_smooth function. Defaults to 8.

    amp : float, optional
        Optional multiplier on the input RS that scales the final RS. The value
        is required to be tuned to ensure the final piecewise loglinear
        function bounds the broadbanded input RS at all points.

        Defaults to 1.02.
            * amp > 1 : the final RS will be larger than the input RS
            * amp < 1 : the final RS will be smaller than the broad-banded
                        input RS.
            * amp = 1 : the final RS will be fitted to match the broad-banded
                        RS as closely as possible

    Returns
    ----------

    fit_frq : ndarray of float
        Frequency values of the final simplified and broadbanded RS

    fit_rs : ndarray of float
        Spectral acceleration values of the final simplified and broadbanded RS

    """
    f_pts = np.logspace(-1, 3, npts, endpoint=True)
    rs_pts = np.ones(len(f_pts))
    params = np.concatenate([f_pts, rs_pts], axis=0)

    bb_frq, bb_rs = broadband_smooth(
            frq, rs, ratio=ratio, window=window, amp=amp)

    popt, pconv = curve_fit(piecewise_logxlinear, bb_frq, bb_rs, p0=params)

    fit_frq = popt[:len(popt)//2]
    fit_rs = popt[len(popt)//2:]

    return fit_frq, fit_rs


def optimal_broadband(frq, rs, npts=8, ratio=0.15,
                      f_pts=None, offset=0, iprint=1):
    """Generate a simplified and broadbanded response spectrum
    from a given response spectrum (RS).

    Parameters
    ----------

    frq : 1D list/tuple/ndarray
        Frequencies of input response spectrum to be broadbanded.

    rs: 1D list/tuple/ndarray
        Spectral acceleration values of input response spectrum
        to be broadbanded. Object should have the same length as frq.

    npts : int, optional
        Positive integer >= 2 that specifies the number of points the
        simplified response spectrum should have. Defaults to 8.

    ratio : float, optional
        Parameter that defines the broadbanding. The input response spectrum
        is shifted by +- ratio. The frequencies in the input RS are multiplied
        by (1+ratio) and (1-ratio). The final simplified, broad-banded
        RS is generated to bound the inputs RS and shifted input RSs.
        The ratio parameter therefore determines how wide the final RS peaks
        should be. Defaults to 0.15 per ASCE 4-98 or Reg. Guide 1.122.

    f_pts : 1D list/tuple/ndarry, optional
        If provided, f_pts defines the list of initial input frequncies and
        total number of points in the final simplified RS. The final optimal RS
        frequncies may deviate from the inputs however. Defaults to None.
        This results in an initial guess of N equally distributed points
        between 0.1Hz and 1000Hz on a log scale; N=npts.

    offset : float, optional
        An offset value in the same units as the spectral acceleration (rs).
        The offset is an optional parameter that defines the minimum gap
        between the final simplified RS and the shifted inputs RSs.The value
        defaults to 0; i.e. the simplified piecewise loglinear RS will bound
        the broadbanded inputs RS as closely as possible.

    iprint : int, optional
        Parameter that defines whether the optimization output is printed.

            * iprint = 1 :  the optimization message is printed
            * iprint != 1 : the function is silent.

    Returns
    ----------

    fit_frq : ndarray of float
        Frequency values of the final simplified and broadbanded RS

    fit_rs : ndarray of float
        Spectral acceleration values of the final simplified and broadbanded RS

    """
    # Generate the initial guess frequencies of the simplified RS
    if f_pts is None:
        npts = max(npts, 8)
        f_pts = np.logspace(-1, 3, npts, endpoint=True)
    else:
        npts = len(f_pts)

    # Generate the initial guess spectral accelerations of the simplified
    # RS by choosing the first >= frq values from the input rs.
    args = list(map(lambda x: np.argwhere(frq >= x)[0][0], f_pts))
    rs_pts = np.array(rs)[args]

    # Define the guess parameters for the optimization routine
    params = np.concatenate([f_pts, rs_pts], axis=0)

    # Broadband the input rs
    bb_frq, bb_rs = broadband(frq, rs, ratio=ratio)

    # Define a minimum gap parameter to keep the final frequencies in the
    # simplified RS from beign too close to each other.
    min_log_gap = 4/(npts-1)/5

    # Define the set of constraints for the optimization routine.
    ineq_const = (
        # The final piecewise loglinear must bound teh broadbanded input RS
        # by at least the offset value:
        {'type': 'ineq', 'fun':
            lambda x: piecewise_logxlinear(bb_frq, *x) - bb_rs - offset},
        # The frequencies of the points in the final RS must not be too
        # close to each other:
        {'type': 'ineq', 'fun':
            lambda x: np.log10(x[1:npts]) - np.log10(x[0:npts-1]) -
            min_log_gap},
        # The minimum frequency should be 0.1Hz, and the max. frequency
        # should be 1000 Hz.
        {'type': 'eq', 'fun':
            lambda x: np.array([x[0] - 0.1, x[npts-1] - 1000])},
        # The simplified RS should not exceed the input RS too much:
        {'type': 'ineq', 'fun':
            lambda x: (max(bb_rs) + offset)*1.02 -
            max(piecewise_logxlinear(bb_frq, *x))}
    )

    # Define the bounds for the frequencies and spectral accelerations:
    bounds = tuple([(0.1, 1000)]*npts + [(0, np.inf)]*npts)

    # Define the maximum number of iterations the optimization routine
    # should try
    options = {'maxiter': 1000}

    # Find the optimum, simplified, broadbanded RS by minimizing the
    # error function. The error function is defined as the square root
    # sum of the squares (SRSS) of the deviation in the final RS
    # and the input broadbanded RS.
    optim_res = minimize(
            error_func, params, args=(bb_frq, bb_rs),
            constraints=ineq_const, bounds=bounds, options=options
            )

    # Print the optimization result
    if iprint == 1:
        print(optim_res.message)

    # Get the optimization result, filter it for the final
    # frequencies and spectral accelerations, return the result.
    popt = optim_res.x

    fit_frq = popt[:len(popt)//2]
    fit_rs = popt[len(popt)//2:]

    return fit_frq, fit_rs


def iter_optimal_broadband(frq, rs, init_npts=8, maxiter=5,
                           ratio=0.15, offset=0, lambda_reg=0.02,
                           iprint=0):
    """Generate a simplified and broadbanded response spectrum
    from a given response spectrum (RS). Uses an iterative approach
    with a regularization error parameter that leads to better fitting
    results than 'optimal_broadband(...)' with fewer unneeded points.

    The iterative optimal broadband function finds a simplified RS for
    the input RS with an initial number of points. The function then
    searches for a new optimum RS by adding a frequency point in the region
    with the highest deviation (error) from the input RS. The step is
    reperformed for 'maxiter' iterations. The 'best' final simplified RS
    is chosen by comparing all the performed iterations. A regularization
    parameter 'lambda_reg' is used to penalize a higher number of input
    points.

    Parameters
    ----------

    frq : 1D list/tuple/ndarray
        Frequencies of input response spectrum to be broadbanded.

    rs: 1D list/tuple/ndarray
        Spectral acceleration values of input response spectrum
        to be broadbanded. Object should have the same length as frq.

   init_npts : int, optional
        Positive integer >= 2 that specifies the minimum number of points
        the simplified response spectrum should have. Defaults to 8.

    maxiter : int, optional
        Positive integer >= 1 that defines the number of iterations of
        optimal broadband to run. Defaults to 5.

    ratio : float, optional
        Parameter that defines the broadbanding. The input response spectrum
        is shifted by +- ratio. The frequencies in the input RS are multiplied
        by (1+ratio) and (1-ratio). The final simplified, broad-banded
        RS is generated to bound the inputs RS and shifted input RSs.
        The ratio parameter therefore determines how wide the final RS peaks
        should be. Defaults to 0.15 per ASCE 4-98 or Reg. Guide 1.122.

    offset : float, optional
        An offset value in the same units as the spectral acceleration (rs).
        The offset is an optional parameter that defines the minimum gap
        between the final simplified RS and the shifted inputs RSs.The value
        defaults to 0; i.e. the simplified piecewise loglinear RS will bound
        the broadbanded inputs RS as closely as possible.

    iprint : int, optional
        Parameter that defines whether the optimization output is printed.

            * iprint = 1 :  the optimization message is printed
            * iprint != 1 : the function is silent.

    Returns
    ----------

    fit_frq : ndarray of float
        Frequency values of the final simplified and broadbanded RS

    fit_rs : ndarray of float
        Spectral acceleration values of the final simplified and broadbanded RS

    """
    # Define the number of initial guess frequencies in four regions:
    # 0.1Hz - 1Hz, 1Hz - 10Hz, 10Hz - 100Hz, and 100Hz - 1000Hz
    # At least 2 points per regions, so at least 8 total,
    regional_npts = [init_npts//4]*4
    regional_npts = np.maximum(regional_npts, [2]*4)

    # Get the initial guess frequencies from the given regional pts
    f_pts = get_init_fpts(regional_npts)

    frq = np.array(frq)
    rs = np.array(rs)

    # Get a smoothened, broadbanded version of the input RS
    # Used to calculate regional errors later
    target_frq, target_rs = broadband_smooth(frq, rs)

    # Define the extents of the regions; the numbers are powers
    # of 10.
    regions = [(-1.2, 0), (0, 1), (1, 2), (2, 3.2)]

    # Filter to separate target_frq, target_rs into the
    # above regions
    def mask(x, i, j):
        return x[(10**i <= target_frq) & (target_frq < 10**j)]

    target_frq_regions = [mask(target_frq, i, j) for i, j in regions]
    target_rs_regions = [mask(target_rs, i, j) for i, j in regions]

    # Dictionary to track results from
    # iterations of the optimization routine
    track_popt = {}

    # Iteration loop
    for i in range(maxiter):

        # Get the optimal simplified RS given the specified
        # initial guess f_pts
        fit_frq, fit_rs = optimal_broadband(
                frq, rs, f_pts=f_pts, ratio=ratio,
                offset=offset, iprint=iprint,
                )

        # Calculate average regional errors of the simplified RS
        # The average is used since each region has a different
        # number of frequencies
        popt = np.concatenate([fit_frq, fit_rs])
        regional_errors = [
                error_func(popt, tfrq, trs)/len(tfrq)
                for tfrq, trs
                in zip(target_frq_regions, target_rs_regions)
                ]

        # Update tracking dictionary
        track_popt[i] = {
                'frq': fit_frq,
                'rs': fit_rs,
                'npts': regional_npts,
                'regional_error': regional_errors,
                'total_error': error_func(popt, frq, rs),
                }

        # Define regularization error that penalizes the number of points
        npts = len(fit_frq)
        init_error = track_popt[0]['total_error']
        track_popt[i]['reg_total_error'] = track_popt[i]['total_error'] + \
            lambda_reg*init_error*npts

        # Add a new frequency point in the region of max. error.
        # Update the initial guess frequencies
        # This is used in the next iteration, if there is one.
        regional_npts[np.argmax(regional_errors)] += 1

        f_pts = get_init_fpts(regional_npts)

    # Find the optimal iteration based on the regularized error
    i_opt = np.argmin([
            track_popt[i]['reg_total_error']
            for i in range(maxiter)
            ])

    # Get the optimal simplified broadbanded RS and return
    # the results.
    fit_frq = track_popt[i_opt]['frq']
    fit_rs = track_popt[i_opt]['rs']

    return fit_frq, fit_rs


# %% Response Spectrum Post-processing - General Utility Functions


def broadband(frq, rs, ratio=0.15):
    """Broadband the input RS by shifting the frequencies
    by +- ratio and +- ratio/2. Concatenate and sort the final five
    RSs, then return the result. The final result looks like a
    broadened noisy version of the initial RS.

    Parameters
    ----------
    frq : 1D list/tuple/ndarray
        Frequencies of input response spectrum to be broadbanded.

    rs: 1D list/tuple/ndarray
        Spectral acceleration values of input response spectrum
        to be broadbanded. Object should have the same length as frq.

    ratio : float, optional
        Parameter that defines the broadbanding. The input response spectrum
        is shifted by +- ratio. The ratio parameter determines how wide the
        final RS peaks should be. Defaults to 0.15 per ASCE 4-98 or
        Reg. Guide 1.122.

    Returns
    ----------

    sorted_frq : ndarray of float
        Frequency values of the sorted, broadbanded RS

    sorted_rs : ndarray of float
        Spectral acceleration values of the sorted, broadbanded RS

    """
    ratios = [1, 1+ratio/2, 1+ratio, 1-ratio/2, 1-ratio]
    frqs = [np.array(frq)*ratio for ratio in ratios]
    rss = [rs]*len(frqs)

    new_frq = np.concatenate(frqs, axis=0)
    new_rs = np.concatenate(rss, axis=0)

    argsort_frq = np.argsort(new_frq)

    sorted_frq = new_frq[argsort_frq]
    sorted_rs = new_rs[argsort_frq]

    return sorted_frq, sorted_rs


def rolling_max(frq, rs, window=8):
    """Return the rolling maximum of the spectral
    acceleration (rs) based on the specified window.
    The window is centered at each frequency (x-value).

    Parameters
    ----------
    frq : 1D list/tuple/ndarray
        Frequencies (x-values) of input.

    rs: 1D list/tuple/ndarray
        Spectral acceleration (y-values) of input.
        Object should have the same length as frq.

    window : int, optional
        Window size for the rolling function. Defaults to 8. An
        odd number is converted to an even number by window - 1.
        The window size is defined by the number of points. It is
        not a frequency range; i.e. it is unitless.

    Returns
    ----------

    frq : ndarray of float
        Frequency values; same as input frq.

    new_rs : ndarray of float
        Rollwing max spectral acceleration values.

    """
    window = 2*(window//2)
    new_frq = frq[window//2:-window//2]
    new_rs = []

    for i in range(window//2, len(rs)-window//2):
        new_rs.append(max(rs[i-window//2:i+window//2]))

    return np.array(new_frq), np.array(new_rs)


def rolling_mean(frq, rs, window=8):
    """Return the rolling mean of the spectral
    acceleration (rs) based on the specified window.
    The window is centered at each frequency (x-value).

    Parameters
    ----------
    frq : 1D list/tuple/ndarray
        Frequencies (x-values) of input.

    rs: 1D list/tuple/ndarray
        Spectral acceleration (y-values) of input.
        Object should have the same length as frq.

    window : int, optional
        Window size for the rolling function. Defaults to 8. An
        odd number is converted to an even number by window - 1.
        The window size is defined by the number of points. It is
        not a frequency range; i.e. it is unitless.

    Returns
    ----------

    frq : ndarray of float
        Frequency values; same as input frq.

    new_rs : ndarray of float
        Rolling mean spectral acceleration values.

    """
    window = 2*(window//2)
    new_frq = frq[window//2:-window//2]
    new_rs = []

    for i in range(window//2, len(rs)-window//2):
        new_rs.append(np.mean(rs[i-window//2:i+window//2]))

    return np.array(new_frq), np.array(new_rs)


def broadband_step(frq, rs, ratio=0.15, window=8):
    """Apply the rolling_max and broadband functions to the input
    frq and rs. Returns the resulting frq and rs.
    """
    bb_frq, bb_rs = broadband(frq, rs, ratio=ratio)
    max_frq, max_rs = rolling_max(bb_frq, bb_rs, window=window)

    return np.array(max_frq), np.array(max_rs)


def broadband_smooth(frq, rs, ratio=0.15, window=8, amp=1.02):
    """Get a smoothened, broadbanded version of the input RS.
    Applies the broadband_step and rolling_mean functions to the RS.
    Returns the resulting frq and rs.

    Parameters
    ----------
    frq : 1D list/tuple/ndarray
        Frequencies of input.

    rs: 1D list/tuple/ndarray
        Spectral acceleration of input.
        Object should have the same length as frq.

    ratio : float, optional
        Parameter that defines the broadbanding. The input response spectrum
        is shifted by +- ratio. The ratio parameter determines how wide the
        final RS peaks should be. Defaults to 0.15 per ASCE 4-98 or
        Reg. Guide 1.122.

    window : int, optional
        Window size for the rolling function. Defaults to 8. An
        odd number is converted to an even number by window - 1.
        The window size is defined by the number of points. It is
        not a frequency range; i.e. it is unitless.

    amp : float, optional
        Optional multiplier on the input RS that scales the final RS.
        Defaults to 1.02.

    Returns
    ----------

    final_frq : ndarray of float
        Frequency values of broadbanded RS.

    final_new_rs : ndarray of float
        Spectral acceleration values of broadbanded RS.

    """
    max_frq, max_rs = broadband_step(frq, rs, ratio=0.15, window=window)

    final_frq, final_rs = rolling_mean(
            max_frq, np.array(max_rs)*amp, window=window)

    return np.array(final_frq), np.array(final_rs)


def piecewise_logxlinear(x, *params):
    """Transform the input frequency, x, into a spectral
    acceleration by interpolating between the RS defined by
    params. The interpolation is loglinear on the frequency
    axis, and linear in the spectral acceleration axis.

    Params defines the points for the piecewise
    loglinear relationship.

    Parameters
    ----------
    x: array_like
        A point or list of points to evaluate the piecewise loglinear
        function.

    params: Variable length argument list
        Defines the input RS to be interpolated. The input should
        list the frequencies of all points, followed by the spectral
        accelerations. Eg:

            * piecewise_logxlinear(x, frq1, frq2, frq3 ...,
                             frq6, rs1, rs2, rs3 ..., rs6)

    Returns
    --------

    y : array_like
        Interpolated/extrapolated spectral acceleration values
        of the input, x, within the RS defined by params.

    """
    xdata = params[:len(params)//2]
    ydata = params[len(params)//2:]
    fit = interp1d(np.log(xdata), ydata,
                   kind='linear', fill_value='extrapolate')

    return fit(np.log(x))


def distance_btw_pts(params):
    """Find the distance between points in a RS where
    the x-axis is on a log-scale and the y-axis is on
    a regular linear-scale.

    Parameters
    ----------
    params : 1D list/tuple/ndarray
        Defines the RS points. Should  have an even length. The list
        contains the frequencies (x-values) first followed by the
        spectral accelerations (y-values). Eg. [x1,x2,x3,y1,y2,y3]

    Returns
    -------
    dist : 1D ndarray
        An array of floats with the distances between the input
        points. The array length is half the length of params.

    """
    x = np.log10(params[:len(params)//2])
    x = (x - x.min())/(x.max() - x.min())

    y = params[len(params)//2:]
    y = (y - y.min())/(y.max() - y.min())

    pts = np.vstack((x, y)).T
    dist = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    return dist


def gradient_btw_pts(params):
    """Find the gradient between points in a RS where
    the x-axis is on a log-scale and the y-axis is on
    a regular linear-scale. The x and y values are
    normalized between 0 and 1 prior to the gradient
    calculation.

    Parameters
    ----------
    params : 1D list/tuple/ndarray
        Defines the RS points. Should  have an even length. The list
        contains the frequencies (x-values) first followed by the
        spectral accelerations (y-values). Eg. [x1,x2,x3,y1,y2,y3]

    Returns
    -------
    grad : 1D ndarray
        An array of floats with the gradients between the input
        points. The array length is half the length of params.

    """
    x = np.log10(params[:len(params)//2])
    x = (x - x.min())/(x.max() - x.min())

    y = np.array(params[len(params)//2:])
    y = (y - y.min())/(y.max() - y.min())

    grad = np.diff(y)/np.diff(x)
    return grad


def error_func(params, frq, rs):
    """Calculate the square root, sum of the squares (SRSS)
    error between a piecewise loglinear RS defined  by
    params, and a target RS defined by frq and rs.

    Parameters
    ----------
    params : Variable length argument list/tuple/ndarray
        Defines the a piecewise loglinear input RS. The inputs is a
        list of the frequencies of all points, followed by the spectral
        accelerations. Eg:[frq1, frq2, frq3, rs1, rs2, rs3]. See
        piecewise_logxlinear(...) for more information.

    frq : 1D, array_like
        Frequencies of the target RS.

    rs : 1D, array_like
        Spectral accelerations of the target RS. Should have the same
        length as frq.

    Returns
    --------
    error : float
        A float that is equal the SRSS of the error between the
        piecewise loglinear approximation, and the target rs.

    """
    est_rs = piecewise_logxlinear(frq, *params)
    error = np.sum((est_rs - rs)**2)**0.5
#     error = np.sum(np.absolute(est_rs - rs))
    return error


def get_init_fpts(regional_npts):
    """Convert a list/tuple of 4 integers to an ndarray of frequenices.
    The 4 numbers define the number of points between 0.1Hz-1Hz, 1Hz-10Hz,
    10Hz-100Hz, and 100Hz-1000Hz.

    """
    regional_f_pts = [np.logspace(exp, exp+1, regional_npts[i], endpoint=False)
                      for i, exp in enumerate(range(-1, 2))]
    regional_f_pts.append(np.logspace(2, 3, regional_npts[-1], endpoint=True))
    f_pts = np.concatenate(regional_f_pts)
    return f_pts
