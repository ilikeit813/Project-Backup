#script of Python script to dedisperse a file, save each DM trials time series

from mpi4py import MPI
import sys
import numpy as np
import glob
#import multiprocessing as mp
import os
import time

def  dedisp(files,pol,DMstart,DMend,freq,tInt):
    """
    Perform a single dispersed pulse search over a range of DMs.
    Required:
    files   -  a spectrum files such as files="spec01.npy",
    spec    -  Spectrogram to be searched.  Must be freq X time.
    freq    -  Array of frequencies in MHz.
    DMstart -  Starting DM in pc cm-3.
    DMend   -  Maximum DM to search.
    tInt    -  Temporal resolution of Spectrogram in seconds.
    RFI mitigation
    Bandpass
    frequency cut off at high and low end
    """
    #freqency cut off at hign and low-> fch, fcl
    fcl=360
    fch=3700
    freq=freq[fcl:fch]/10**6
    cent_freq = np.median(freq)
    dDM= 1.0/len(freq)
    nDMs = DMs(DMstart,DMend,dDM)
    BW   = freq.max()-freq.min()
    #path=os.getcwd()
    #ininate the dedispsioin timeseries bank
    
    
    RFI=np.load('./rfi/rfiflag_%.46s_pol_%.1i_fbins.npy' % (files,pol))-fcl
    bandpas=(np.load('./bp/bandpass_%.16s_pol_%.1i.npy' % (files,pol))[fcl:fch])
    #bandpas=(np.load('./bp/bandpass_%.16s_pol_%.1i.npy' % (files,pol))[fcl:fch]).astype(np.float16)
    spec=(np.load(files,mmap_mode='r')[:,pol,fcl:fch])/bandpas
    #spec=(np.load(files,mmap_mode='r')[:,pol,fcl:fch]).astype(np.float16)/bandpas
    #print spec.shape
    #ats=np.zeros(0)
    #t4=time.time()
    #print 'spec.dtype=',spec.dtype
    #spec_shape  = spec.shape[0]
    #spec_shape0 = spec_shape*lenfiles
    #mask the rfi in this file with bandpass
    spec[:,RFI]=1.0



    for i in range(1,150/processes_use):
        files=fn[rank*150/processes_use + i]
        RFI=np.load('./rfi/rfiflag_%.46s_pol_%.1i_fbins.npy' % (files,pol))-fcl
        speci=(np.load(files,mmap_mode='r')[:,pol,fcl:fch])/bandpas
        speci[:,RFI]=1.0
        spec=np.append(spec,speci,0)
        #if rank==0:
        #print 'i',i,'rank',rank,'speci',speci[0:5,0],'RFI',RFI[0:10] 

    spec_shape  = spec.shape[0]
    spec_shape0 = spec_shape*lenfiles

    """
    if rank==0:
        print 'spec_shape',spec_shape,'10000x2'
        print 'spec_shape0',spec_shape0,'1500000'
    """

    DM=DMstart
    dmskip=0
    dmcounter=0


    if rank<15:
        npws = int(np.round(np.log2(maxpw/tInt)))
        BW   = freq.max()-freq.min()
        fInt = BW/len(freq)*1e3
        spec_std='N/A'
        version=3
        inFile='N/A'
        pulseno=0
        fileno=1
        filename = "pp_SNR_pol_%.1i_td_%.2i_No_%05d.txt" % (pol,rank,fileno)
        outfile = open(filename,'w')



    #"""


    #t5=time.time()
    for iDM in range(dmskip,nDMs):
        tb=np.round((delay2(freq,DM)/tInt)).astype(np.int32)
        ts=np.zeros((tb.max()+spec.shape[0]))
        for freqbin in range(0,len(freq)):
            ts[tb.max()-tb[freqbin]:tb.max()-tb[freqbin]+spec.shape[0]]+=spec[:,freqbin]
            #ats=np.append(np.append(ats,ts),np.nan)#store to bank

        tsi=np.zeros(spec_shape0+tb.max())
        tsi[rank*spec_shape:rank*spec_shape+len(ts)]=ts
        tstotal=tsi*0
        comm.Allreduce(tsi,tstotal,op=MPI.SUM)

        if rank<15:
                if pulseno >10000*fileno:
                    outfile.close()
                    fileno+=1
                    filename = "pp_SNR_pol_%.1i_td_%.2i_No_%05d.txt" % (pol,rank,fileno)
                    outfile = open(filename,'w')

                #npws = int(np.round(np.log2(maxpw/tInt)))
                #BW   = freq.max()-freq.min()
                #fInt = BW/len(freq)*1e3
                #spec_std='N/A'
                #version=3
                #inFile='N/A'
                #pulseno=0
                #header = #This is SDPS for LWA1 version {0}
                #on dataset: {1}.  Spectrogram resolution: {7} kHz X {8} ms. Spectrogram RMS = {2}(averaged).
                #Spectrogram Bandwidth = {9} MHz.  Spectrogram temporal length = {10} s.
                #{3} DM's were searched between {4} and {5} pc cm-3 with spacing dDM = {6}.
                #Number of points in a time series (=length of time series)
                #
                #Pulse #     SNR             DM           time            dtau         dnu         nu        mean       rms
                #                         (pc cm-3)        (s)            (s)        (kHz)       (MHz)
                #==============================================================================================================
                #filename = "SNR_pol_%.1i.txt" % pol
                #outfile = open(filename,'w')
                #outfile.write(header.format(version,inFile,spec_std, nDMs,DMstart,DMend,dDM,fInt,tInt,BW,spec_shape0*tInt))
                #outfile.write('\n')

                #DMarray=np.zeros(4)
                #DMarray[0]=iDM
                #DMarray[1]=DM
                #DMarray[2]=t4-t3
                #DMarray[3]=time.time()-t5
                #np.savetxt('i%.1i' % pol,DMarray)

                tst = int(np.round(len(tstotal)-spec_shape0))
                tstotal = tstotal[tst:len(tstotal)-tst]
                #decimate the time series
                ndown = 2**rank
                #for tw in range(npws):
                    #t7=time.time()
                    #print 'time widtn=',tw
                    #print 'tstotal.shape=',tstotal.shape
                    #print 'mean=',np.mean(tstotal)
                    #print 'rms=',np.std(tstotal)
                    #print np.where( (tstotal-np.mean(tstotal))/np.std(tstotal) > 5 )
                sn,mean,rms = Threshold(Decimate_ts(tstotal,ndown),thresh,niter=8)
                    #t8=time.time()
                    #print 'iDM,tw,mean,rms=',iDM,tw,mean,rms
                ones = np.where(sn!=-1)[0]
                    #t9=time.time()
                    # Now record all pulses above threshold
                for one in ones:
                        pulse = OutputSource()
                        pulseno += 1
                        pulse.pulse = pulseno
                        pulse.SNR = sn[one]
                        pulse.DM = DM
                        pulse.time = one*tInt*ndown
                        pulse.dtau = tInt*ndown
                        pulse.dnu = fInt
                        pulse.nu = cent_freq
                        pulse.mean = mean
                        pulse.rms = rms
                        outfile.write(pulse.formatter.format(pulse)[:-1]) 
                #ndown*=2 # End of decimaion loop  
                
                

        DM+=dDM*DM   # End of DM loop
        #dmcounter+=1
        #checkatsfinal=1
        #if (ats.shape[0] > 10**8) or (dmcounter>1000):
        #    np.save('./p%.1i/%.5i_%s' % (pol,iDM,files[30:35]),ats)
        #    ats=np.zeros(0)
        #    checkatsfinal=0
        #    dmcounter=0
    #if checkatsfinal!=0:
        #np.save('./p%.1i/%.5i_%s' % (pol,iDM,files[30:35]),ats)


    #"""


    return

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

    for i in range(niter):
        ones = np.where((ts-mean)/std < clip)[0]  # only keep the values less than 3sigma
        mean = np.mean(ts[ones])
        std  = np.std(ts[ones])
        #print mean, std
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
      nu    = None      # Central Observing Frequency
      mean  = None      # Mean in the time series
      rms   = None      # RMS in the time series

      formatter = "{0.pulse:07d}    {0.SNR:10.6f}     {0.DM:10.4f}     {0.time:10.6f} "+\
                 "     {0.dtau:10.6f}     {0.dnu:.4f}     {0.nu:.4f}    {0.mean:.5f}"+\
                 "    {0.rms:0.5f}\n " 

      def __str__(self):
          return self.formatter.format(self)

def ff(x):
    if pol==0 or pol==1:
        fre=np.load('freq1.npy')
        return dedisp(fn[x*150/processes_use],pol,DMstart=ds,DMend=de,freq=fre,tInt=t)

    else: 
        fre=np.load('freq2.npy')
        return dedisp(fn[x*150/processes_use],pol,DMstart=ds,DMend=de,freq=fre,tInt=t)

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    node_use=5
    processes_use=node_use*15
    lenfiles=processes_use

    maxpw=1 #Maximum pulse width to search in seconds. default = 1 s.
    thresh=5

    fn=sorted(glob.glob('056*.npy'))
    t=np.load('tint.npy')
    pol=1
    #if not os.path.exists('p%.1i' % pol):
    #    os.makedirs('p%.1i' % pol)
    ds=1.0#1.0 #initial DM trial
    de=90.0#90.0 #finial  DM trial

    ff(rank)
