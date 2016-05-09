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

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    fpp=2#spectrogram per processer
    nodes=30
    pps=10
    numberofFiles=fpp*nodes*pps

    maxpw=10 #Maximum pulse width to search in seconds. default = 1 s.
    thresh=5.0#SNR cut off

    fn=sorted(glob.glob('056*.npy')) 
    tInt=np.load('tInt.npy')

    DMstart=6.0#1.0 #initial DM trial
    DMend  =16.0#90.0 #finial  DM trial

    fcl=360#low frequency cut off
    fch=3700#high frequency cut off

    npws = int(np.round(np.log2(maxpw/tInt)))+1 # +1 Due to in range(y) it goes to y-1 only

    spect=np.append(np.asarray((np.load(fn[rank*fpp],mmap_mode='r')),dtype=np.float64), np.asarray((np.load(fn[rank*fpp+1],mmap_mode='r')),dtype=np.float64),0)

    #bp=np.zeros((4,4095))
    for pol in (0,1,2,3):
     for i in range(3):
        bp=spect[:,pol,:].mean(0)
        bptotal=bp*0
        comm.Allreduce(bp,bptotal,op=MPI.SUM)#merge the 4 bandpass from all processor
        bptotal/=size
        test=smoth.smooth(bptotal,window_len=41)
        for ii in range(3):
                a=bptotal/test
                for iii in np.where(a>1):
                    bptotal[iii]=test[iii]
                test=smoth.smooth(bptotal,window_len=101)
        bp=test
        np.save('bandpassitpol%.1d_%.1d' % (pol,i),bp)


        """
        bp[pol,:]=spect[:,pol,:].mean(0)#calculate bandpass
        bptotal=bp*1#initiate bp
        test=smoth.smooth(bptotal[pol,:],window_len=41)
        for ii in range(3):
                a=bptotal[pol,:]/test
                for i in np.where(a>1):
                    bptotal[pol,i]=test[i]
                test=smoth.smooth(bptotal[pol,:],window_len=101)
        bp[pol,:]=test
        spect[:,pol,:]/=bp[pol,:]#bandpass removal
        """
        spect[:,pol,:]/=bp#bandpass removal

    for pol in range(4):
            bp=np.zeros((fch-fcl))
            bpmean=spect[:,pol,fcl:fch].mean(0)
            comm.Allreduce(bpmean,bp,op=MPI.SUM)#merge the 4 bandpass from all processor
            bp/=size
            bpvasize=np.zeros((size))
            bpvasize[rank] = ( (spect[:,pol,fcl:fch].mean(0) - bp.mean() )**2 ).mean()
            bpvariance=np.zeros((size))            
            comm.Allreduce(bpvasize,bpvariance,op=MPI.SUM)#merge the 4 bandpass from all processor
            bpvariance = bpvariance.mean()
            if rank==0:
                print 'pol',pol,'mean of bandpass',bp.mean()
                #print 'pol',pol,'vari of bandpass',bpvariance
                print 'pol',pol,'Standard Variation of bandpass',bpvariance**0.5
                np.save('bp_pol%.1d' % pol, bp)

            ts=np.zeros((numberofFiles*np.load(fn[0],mmap_mode='r').shape[0]))
            tstotal=ts*0#initiate a 4 hour blank time series
            ts[rank*spect.shape[0]:rank*spect.shape[0]+spect.shape[0]]+=(spect[:,pol,fcl:fch]).mean(1)
            comm.Allreduce(ts,tstotal,op=MPI.SUM)#merge the 4 hour timeseries from all processor
            tsvasize=np.zeros((size))
            tsvasize[rank] = ( (spect[:,pol,fcl:fch].mean(1) - tstotal.mean() )**2 ).mean()
            tsvariance=np.zeros((size))
            comm.Allreduce(tsvasize,tsvariance,op=MPI.SUM)#merge the 4 bandpass from all processor
            tsvariance = tsvariance.mean()
            if rank==0:
                print 'pol',pol,'mean of timseries',tstotal.mean()
                #print 'pol',pol,'vari of time series',tsvariance
                print 'pol',pol,'Standard Varia. of timeseries',tsvariance**0.5
                np.save('ts_pol%.1d' % pol, tstotal)
