'''
Created: Nov 2017
Latest update:  May 2019
@author: Praveer Nidamaluri

Module for generating acceleration response spectra.
May 2019 - Added additional functions to test out multiprocessing
'''

import os
from multiprocessing import Pool, cpu_count, Pipe, Process
import time as timer
import numpy as np
from itertools import accumulate
import matplotlib.pyplot as plt


def test_func(k,k2):
    func = lambda x: x*x
    return [os.getpid(),1.5031,k,k2,func(k)]          

def loop(n):
    s = 0
    for i in range(int(n)):
        s+=n
    return [os.getpid(),s]    

def loop_pr(n,send_end):
    s = 0
    for i in range(int(n)):
        s+=n
    send_end.send([os.getpid(),s])
    send_end.close()

def step_resp_single(wn,frq_mult,dt_max,time_a,acc,zeta):
        nyq = max(frq_mult*wn/2/np.pi,20)
        dt = max(1/(2*nyq),dt_max)
        dt_tm = np.arange(0,time_a[-1],dt)
        dt_acc = np.interp(dt_tm,time_a,acc)
        
        # Calculate response acceleration time history
        A,B = get_step_matrix(wn,zeta,dt)
        func = lambda x,a : np.dot(A,x) + np.dot(B,a)        
        act = np.column_stack((dt_acc[:-1],dt_acc[1:]))
        act = np.append(np.array([[0,0],[0,dt_acc[0]]]),act,axis=0)
        x = np.array(list(accumulate(act,func)))
        temp = -np.array([wn**2,2*zeta*wn])
        z = np.dot(x,temp)
        return np.max(np.absolute(z))
    
def step_resp_single_pr(wn,frq_mult,dt_max,time_a,acc,zeta,send_end):
        nyq = max(frq_mult*wn/2/np.pi,20)
        dt = max(1/(2*nyq),dt_max)
        dt_tm = np.arange(0,time_a[-1],dt)
        dt_acc = np.interp(dt_tm,time_a,acc)
        
        # Calculate response acceleration time history
        A,B = get_step_matrix(wn,zeta,dt)
        func = lambda x,a : np.dot(A,x) + np.dot(B,a)        
        act = np.column_stack((dt_acc[:-1],dt_acc[1:]))
        act = np.append(np.array([[0,0],[0,dt_acc[0]]]),act,axis=0)
        x = np.array(list(accumulate(act,func)))
        temp = -np.array([wn**2,2*zeta*wn])
        z = np.dot(x,temp)
        send_end.send(np.max(np.absolute(z)))
        send_end.close()

def step_resp_spect_mpl(acc,time_a,zeta=0.05,ext=True,plot=False,max_nyq=500,accuracy = 0.2,ncpu=-1,chunksize=None):
    '''
    Only accurate up till 200Hz or so - for more accuracy must manually change the interpolation step/nyquist 
    freq.
    Input: acceleration array, time array of same length of acc., critical damping ratio in 
    decimal, ext = boolean (if true, frequencies between 100Hz and 1000Hz are also included), plot = present 
    or not, accuracy = between 0, and 1 (1 = most accurate, but slow, 0 = less accurate but fast)
    ncpu = number of cpu processes to use (-1 = all cpus (default), otherwise an integer number of cpus)
    Output: array with response in g's, array with frequencies (in Hz)
    Also prints time taken, and displays output spectrum
    
    Adaptation of method described in "Calculation of response Sepctra from mMStong-Motion Earthquake Records"
    Nigam, Jennings, Bulletin of the Seismological Society of America vol 59, no. 2, April 1969.
    
    '''

    t0 = timer.clock()
       
    # Set up  correct ncpu value
    
    if ncpu == -1:
        ncpu = cpu_count()
    else:
        ncpu = max(min(int(ncpu),cpu_count()),1)
    
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
    
    dt_max = 1/(2*max_nyq)
    
    # Calculate response for a spring with each wn
    args = (frq_mult,dt_max,time_a,acc,zeta)
    with Pool(ncpu) as p:
#        rs = p.starmap_async(step_resp_single,[(wn,*args) for wn in w],chunksize=1).get()
        rs = p.starmap(step_resp_single,[(wn,*args) for wn in w],chunksize=chunksize)
    
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


def step_resp_spect_mpr(acc,time_a,zeta=0.05,ext=True,plot=False,max_nyq=500,accuracy = 0.2,ncpu=-1):
    '''
    Only accurate up till 200Hz or so - for more accuracy must manually change the interpolation step/nyquist 
    freq.
    Input: acceleration array, time array of same length of acc., critical damping ratio in 
    decimal, ext = boolean (if true, frequencies between 100Hz and 1000Hz are also included), plot = present 
    or not, accuracy = between 0, and 1 (1 = most accurate, but slow, 0 = less accurate but fast)
    ncpu = number of cpu processes to use (-1 = all cpus (default), otherwise an integer number of cpus)
    Output: array with response in g's, array with frequencies (in Hz)
    Also prints time taken, and displays output spectrum
    
    Adaptation of method described in "Calculation of response Sepctra from mMStong-Motion Earthquake Records"
    Nigam, Jennings, Bulletin of the Seismological Society of America vol 59, no. 2, April 1969.
    
    '''

    t0 = timer.clock()
       
    # Set up  correct ncpu value
    
    if ncpu == -1:
        ncpu = cpu_count()
    else:
        ncpu = max(min(int(ncpu),cpu_count()),1)
    
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
    
    dt_max = 1/(2*max_nyq)
    
    # Calculate response for a spring with each wn
    args = (frq_mult,dt_max,time_a,acc,zeta)
    
    processes = []
    pipe_list = []
    
    for wn in w:
        recv_end, send_end = Pipe(False)
        p = Process(target=step_resp_single_pr,args=(wn,*args,send_end))
        processes.append(p)
        pipe_list.append(recv_end)
        p.start()
    
    for p in processes:
        p.join()
        
    rs = np.array(list([r.recv() for r in pipe_list]))
    
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
        
    import time as timer
    import numpy as np
    from itertools import accumulate
    import matplotlib.pyplot as plt
    
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


def get_step_matrix(w,zeta=0.05,dt=0.005):
    '''
    For use with the step_resp_spect() function that generates response spectra in the step-by-step method.
    '''
    
    import numpy as np
    
    A = np.zeros((2,2))
    B = np.zeros((2,2))
    
    exp = np.exp(-zeta*w*dt)
    zsqt = (1-zeta**2)**0.5
    sin = np.sin(w*zsqt*dt)
    cos = np.cos(w*zsqt*dt)
    
    A[0,0] = exp*(cos+sin*zeta/zsqt)
    A[0,1] = exp/(w*zsqt)*sin
    A[1,0] = -w/zsqt*exp*sin
    A[1,1] = exp*(cos-sin*zeta/zsqt)
    
    t1 = (2*zeta**2 - 1)/w**2/dt
    t2 = 2*zeta/w**3/dt
    
    B[0,0] = exp*(sin/(w*zsqt)*(t1 + zeta/w) + cos*(t2 + 1/w**2)) - t2
    B[0,1] = -exp*(sin/(w*zsqt)*t1 + cos*t2) - 1/w**2 + t2
    B[1,0] = exp*((t1 + zeta/w)*(cos - sin*zeta/zsqt) - (t2 + 1/w**2)*(sin*w*zsqt + cos*zeta*w)) + 1/w**2/dt
    B[1,1] = -exp*(t1*(cos - sin*zeta/zsqt) - t2*(sin*w*zsqt + cos*zeta*w)) - 1/w**2/dt
    
    return A, B


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
        
    import time as timer
    import numpy as np
    import matplotlib.pyplot as plt
    
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