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
import csv

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


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    fpp=2#spectrogram per processer
    nodes=34
    pps=10
    numberofFiles=fpp*nodes*pps

    maxpw=10 #Maximum pulse width to search in seconds. default = 1 s.
    thresh=5.0#SNR cut off

    fn=sorted(glob.glob('056*.npy')) 
    tInt=np.load('tInt.npy')

    DMstart=1.0#1.0 #initial DM trial
    DMend  =5000.0#90.0 #finial  DM trial

    fcl=360#low frequency cut off
    fch=3700#high frequency cut off

    npws = int(np.round(np.log2(maxpw/tInt)))+1 # +1 Due to in range(y) it goes to y-1 only

    spect=np.append(np.asarray((np.load(fn[rank*fpp],mmap_mode='r')),dtype=np.float64), np.asarray((np.load(fn[rank*fpp+1],mmap_mode='r')),dtype=np.float64),0)
    bp=np.zeros((4,4095))
    for pol in (0,1,2,3):
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

    for pol in range(4):#""" RFI removal and baseline removal
        c11=np.median(spect[:,pol,fcl:fch].mean(0))# replace the RFI along bandpass
        for i in np.where(spect[:,pol,fcl:fch].mean(0)>1.03)[0]:
            spect[:,pol,i+fcl]=c11
        c22=np.median(spect[:,pol,fcl:fch].mean(1))# replace the RFI al ng time series
        for i in np.where(spect[:,pol,fcl:fch].mean(1)>1.03)[0]:
            spect[i,pol,:]=c22

    #pol=0
    #if pol==0:
    for pol in (0,2):
        c=[]#blak record for SNR
        if pol==0:
            freq=np.load('freq1.npy')
        else: 
            freq=np.load('freq2.npy')
        freq=freq[fcl:fch]/10**6
        cent_freq = np.median(freq)
        dDM= 1.0/len(freq)
        nDMs = DMs(DMstart,DMend,dDM)
        BW   = freq.max()-freq.min()
        DM=DMstart

        txtsize=np.zeros((npws,2),dtype=np.int32) #fileno = txtsize[ranki,0], pulse number = txtsize[ranki,1],ranki is the decimated order of 2
        txtsize[:,0]=1 #fileno star from 1

    
        tbmax=0 #repeat control, if dedispersion time series are identical, skip dedispersion calculation
        for iDM in range(0,nDMs):
            tb=np.round((delay2(freq,DM)/tInt)).astype(np.int32)

            if tb.max()-tbmax==0:#identical dedispersion time series checker
                tbmax=tb.max()
                DM+=dDM*DM
                #if rank ==0:
                #    print 'DM',DM,'skipped'
                continue
            tbmax=tb.max()

            ts=np.zeros((tb.max()+numberofFiles*np.load(fn[0],mmap_mode='r').shape[0]))
            for freqbin in range(len(freq)): 
                ts[tb.max()-tb[freqbin]+rank*spect.shape[0]:tb.max()-tb[freqbin]+rank*spect.shape[0]+spect.shape[0]]+=(spect[:,pol,freqbin]+spect[:,pol+1,freqbin])/2.0 
            tstotal=ts*0#initiate a 4 hour blank time series
            comm.Allreduce(ts,tstotal,op=MPI.SUM)#merge the 4 hour timeseries from all processor
            tstotal = tstotal[tb.max():len(tstotal)-tb.max()]#cut the dispersed time lag

            #if rank==0:
            #   print '1',tb.max()*tInt
            #   print '2',tstotal.shape, tstotal.max(), tstotal.min(), tstotal.std()
            #   print 'DM',DM

            if rank<npws:#timeseries is ready for signal search
                ranki=rank
                filename = "pp_SNR_pol_%.1i_td_%.2i_no_%.05i.txt" % (pol,ranki,txtsize[ranki,0])
                outfile = open(filename,'a')
                ndown = 2**ranki #decimate the time series
                sn,mean,rms = Threshold(Decimate_ts(tstotal,ndown),thresh,niter=0)
                ones = np.where(sn!=-1)[0]
                #print '# of SNR>5',ones.shape,'decimated time',ndown
                #"""
                for one in ones:# Now record all pulses above threshold
                    txtsize[ranki,1] += 1
                    c.append((txtsize[ranki,1], sn[one], DM, one*tInt*ndown, tInt*ndown, freq[1]-freq[0], cent_freq, mean, rms))#update the record
                    if txtsize[ranki,1] >100000*txtsize[ranki,0]:
                        csv.writer(outfile,lineterminator='\n').writerows(c)
                        outfile.close()
                        c=[]#new start
                        txtsize[ranki,0]+=1
                        filename = "pp_SNR_pol_%.1i_td_%.2i_no_%.05d.txt" % (pol,ranki,txtsize[ranki,0])
                        outfile = open(filename,'a')
                 #"""
            DM+=dDM*DM # End of DM loop

        if rank <npws:
            csv.writer(outfile,lineterminator='\n').writerows(c)
            outfile.close()
