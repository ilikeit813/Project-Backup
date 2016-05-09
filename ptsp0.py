#combine a pulsar'DM dedispersion time series
import sys
import numpy as np
import glob
import os
import time
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

if __name__ == '__main__':
    rank  =  0
    pol   =  0
    fpp   =  6895 #spectrogram per processer you want, limited mainly by 64GB memory per node
    nodes =  1 #the number of node requensted in sh
    pps   =  1 #processer per node requensted in sh
    numberofFiles=fpp*nodes*pps

    totalnumberofspec = 6895.

    maxpw = 10 #Maximum pulse width to search in seconds. default = 1 s.
    dDM   = 1.0/(3700-360)*10 #1.0/len(freq)
    thresh= 5.0 #SNR cut off

    fn   = sorted(glob.glob('05*.npy')) 
    tInt = np.load('tInt.npy')
    bp0 = np.load('bandpassp0.npy')
    bp1 = np.load('bandpassp1.npy')
    std = 0.725 # judged from all raw data, the medium std

    DMstart= 9.195 #1.0 #initial DM trial
    DMend  = 9.196 #90.0 #finial  DM trial

    fcl = 360#low frequency cut off
    fch =3700#high frequency cut off

    if pol <3:
        if pol==0:
            freq=np.load('freq1.npy')
            bp=bp0
        else: 
            freq=np.load('freq2.npy')
            bp=bp1

        freq=freq[fcl:fch]/10**6
        cent_freq = np.median(freq)
        nDMs = DMs(DMstart,DMend,dDM)
        BW   = freq.max()-freq.min()
        DM=DMstart
    
        for iDM in range(0,1):
            tb=np.round((delay2(freq,DM)/tInt)).astype(np.int32)
            ts=np.zeros((tb.max()+numberofFiles*np.load(fn[0],mmap_mode='r').shape[0]))

            for i in range(fpp):

		spect=np.load(fn[i],mmap_mode='r')[:,pol,fcl:fch]
		print i,'combining'
		spect = spect/bp/np.median(spect/bp)

		#RFI removal
		print 'RFI removing'
		spect[spect > np.median(spect)+3*std] = np.median(spect)

		for freqbin in range(len(freq)): 
			ts[tb.max()-tb[freqbin] + (rank*fpp+i)*spect.shape[0] :tb.max()-tb[freqbin] + (rank*fpp+i+1)*spect.shape[0] ] += spect[:,freqbin]

        np.save('dm9195p%s' % pol, ts)
