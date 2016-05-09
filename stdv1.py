#script of Python script to dedisperse a file, save each DM trials time series
#Codes from Sean, modified by Jamie
#V3, super cluster technique

from mpi4py import MPI
import sys
import numpy as np
import glob
import os
import time
import smoth
import sys

def DMs(DMstart,DMend,dDM):
    """
    Calculate the number of DMs searched between DMstart and DMend, with spacing dDM * DM.

    Required:

    DMstart   - Starting Dispersion Measure in pc cm-3
    DMend     - Ending Dispersion Measure in pc cm-3
    dDM       - DM spacing in pc cm-3
    """

    NDMs = np.log10(float(DMend)/float(DMstart))/np.log10(1.0+dDM)

    return int(np.round(NDMs))


def delay2(freq, dm):
    """
    Calculate the relative delay due to dispersion over a given frequency
    range in Hz for a particular dispersion measure in pc cm^-3.  Return
    the dispersive delay in seconds.  Same as delay, but w.r.t to highest frequency.
    ***Used to simulate a dispersed pulse.***

    Required:

    freq - 1-D array of frequencies in MHz
    dm   - Dispersion Measure in pc cm-3
    """
    # Dispersion constant in MHz^2 s / pc cm^-3
    _D = 4.148808e3
    # Delay in s
    tDelay = dm*_D*((1/freq)**2 - (1/freq.max())**2)

    return tDelay


def Threshold(ts, thresh, clip=3, niter=1):
    """
    Wrapper to scipy threshold a given time series using Scipy's threshold function (in 
    scipy.stats.stats).  First it calculates the mean and rms of the given time series.  It then 
    makes the time series in terms of SNR.  If a given SNR value is less than the threshold, it is 
    set to "-1".  Returns a SNR array with values less than thresh = -1, all other values = SNR.
    Also returns the mean and rms of the timeseries.

    Required:

    ts   -  input time series.

    Options:

    thresh  -  Time series signal-to-noise ratio threshold.  default = 5.
    clip    -  Clipping SNR threshold for values to leave out of a mean/rms calculation.  default = 3.
    niter   -  Number of iterations in mean/rms calculation.  default = 1.

    Usage:
    >>sn, mean, rms = Threshold(ts, *options*)

    """
    #  Calculate, robustly, the mean and rms of the time series.  Any values greater than 3sigma are left
    #  out of the calculation.  This keeps the mean and rms free from sturation due to large deviations.

    mean = np.mean(ts) 
    std  = np.std(ts)  
    #print mean,std

    if niter > 0:
        for i in range(niter):
            ones = np.where((ts-mean)/std < clip)[0]  # only keep the values less than 3sigma
            mean = np.mean(ts[ones])
            std  = np.std(ts[ones])
    SNR = (ts-mean)/std
    SNR[SNR<thresh]=-1
    #SNR = st.threshold((ts-mean)/std, threshmin=thresh, newval=-1)

    return SNR, mean, std

def Decimate_ts(ts, ndown=2):
    """
    Takes a 1-D timeseries and decimates it by a factore of ndown, default = 2.  
    Code adapted from analysis.binarray module: 
      http://www.astro.ucla.edu/~ianc/python/_modules/analysis.html#binarray 
    from Ian's Python Code (http://www.astro.ucla.edu/~ianc/python/index.html)
    
    Optimized for time series' with length = multiple of 2.  Will handle others, though.

    Required:
    
    ts  -  input time series

    Options:
    
    ndown  -  Factor by which to decimate time series. Default = 2.
              if ndown = 1, returns ts       
    """

    if ndown==1:
       return ts

    ncols = len(ts)
    n_rep = ncols / ndown
    ts_ds = np.array([ts[i::ndown][0:n_rep] for i in range(ndown)]).mean(0)

    return ts_ds


class OutputSource():

      pulse = None  # Pulse Number
      SNR   = None  # SNR of pulse
      DM    = None  # DM (pc/cm3) of pulse
      time  = None  # Time at which pulse ocurred
      dtau  = None  # Temporal resolution of time series
      dnu   = None  # Spectral resolution
      nu    = None  # Central Observing Frequency
      mean  = None  # Mean in the time series
      rms   = None  # RMS in the time series

      formatter = "{0.pulse:07d}    {0.SNR:10.6f}     {0.DM:10.4f}     {0.time:10.6f} "+\
                 "     {0.dtau:10.6f}     {0.dnu:.4f}     {0.nu:.4f}    {0.mean:.5f}"+\
                 "    {0.rms:0.5f}\n " 

      def __str__(self):
          return self.formatter.format(self)

def bandpass(sp):
    ws = 151 # smooth windows size
    for pol in range(4):
        for i in range(3):
            a=smoth.smooth(np.median(sp[:,pol,:],0),ws)
            test=np.median(sp[:,pol,:],0)*1.0
            for i in range(1):
                test[np.where(test/a>1)]=a[np.where(test/a>1)]
                a=smoth.smooth(test,ws)
            sp[:,pol,:]/=a
    sp[:,0,:]=np.mean(sp[:,0:2,:],1)
    sp[:,1,:]=np.mean(sp[:,2:4,:],1)
    #for pol in range(2):
    #    sp[:,pol,np.where(sp[:,pol,:].mean(0)>maxf)]=np.median(sp[:,pol,:].mean(0))
    #    sp[np.where(sp[:,pol,:].mean(1)>maxt),pol,:]=np.median(sp[:,pol,:].mean(1))
    return(sp[:,0:2,:])


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    fn = sorted(glob.glob('056*.npy'))
    fcl=360
    fch=3700
    for ii in range(2):
     spect=bandpass(np.asarray((np.load(fn[rank*2+ii],mmap_mode='r')[:,:,fcl:fch]),dtype=np.float64))
     for pol in range(2):
         for i in range(2):
             print fn[rank*2+ii],pol,i,spect[:,pol,:].mean(i).mean(),spect[:,pol,:].mean(i).std(),np.median(spect[:,pol,:].mean(i))
