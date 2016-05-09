"""
$Rev: 993 $
$LastChangedBy: jdowell $
$LastChangedDate: 2012-08-30 19:30:21 -0400 (Thu, 30 Aug 2012) $
"""


import os
import sys
import math
import numpy
import ephem
import getopt

import lsl.reader.drx as drx
import lsl.reader.drsu as drsu
import lsl.reader.errors as errors
import lsl.statistics.robust as robust
import lsl.correlator.fx as fxc
from lsl.astro import unix_to_utcjd, DJD_OFFSET

import matplotlib.pyplot as plt


def usage(exitCode=None):
	print """drxWaterfallDRSU.py - Read in a DRX file directly from a DRSU and create a 
collection of time-averaged spectra.  These spectra are saved to a NPZ file called 
<filename>-waterfall.npz.

Usage: drxWaterfallDRSU.py [OPTIONS] md_device file_tag

Options:
-h, --help                  Display this help information
-t, --bartlett              Apply a Bartlett window to the data
-b, --blackman              Apply a Blackman window to the data
-n, --hanning               Apply a Hanning window to the data
-s, --skip                  Skip the specified number of seconds at the beginning
                            of the file (default = 0)
-a, --average               Number of seconds of data to average for spectra 
                            (default = 1)
-d, --duration              Number of seconds to calculate the waterfall for 
                            (default = 10)
-q, --quiet                 Run drxSpectra in silent mode and do not show the plots
-l, --fft-length            Set FFT length (default = 4096)
-c, --clip-level            FFT blanking clipping level in counts (default = 0, 
                            0 disables)
-e, --estimate-clip         Use robust statistics to estimate an approprite clip 
                            level (overrides the `-c` option)
-o, --output                Output file name for waterfall image
"""

	if exitCode is not None:
		sys.exit(exitCode)
	else:
		return True


def parseOptions(args):
	config = {}
	# Command line flags - default values
	config['offset'] = 0.0
	config['average'] = 1.0
	config['LFFT'] = 4096
	config['freq1'] = 0
	config['freq2'] = 0
	config['maxFrames'] = 28000
	config['window'] = fxc.noWindow
	config['output'] = None
	config['duration'] = 10.0
	config['verbose'] = True
	config['clip'] = 0
	config['estimate'] = False
	config['args'] = []

	# Read in and process the command line flags
	try:
		opts, args = getopt.getopt(args, "hqtbnl:o:s:a:d:c:e", ["help", "quiet", "bartlett", "blackman", "hanning", "fft-length=", "output=", "skip=", "average=", "duration=", "clip-level=", "estimate-clip"])
	except getopt.GetoptError, err:
		# Print help information and exit:
		print str(err) # will print something like "option -a not recognized"
		usage(exitCode=2)
	
	# Work through opts
	for opt, value in opts:
		if opt in ('-h', '--help'):
			usage(exitCode=0)
		elif opt in ('-q', '--quiet'):
			config['verbose'] = False
		elif opt in ('-t', '--bartlett'):
			config['window'] = numpy.bartlett
		elif opt in ('-b', '--blackman'):
			config['window'] = numpy.blackman
		elif opt in ('-n', '--hanning'):
			config['window'] = numpy.hanning
		elif opt in ('-l', '--fft-length'):
			config['LFFT'] = int(value)
		elif opt in ('-o', '--output'):
			config['output'] = value
		elif opt in ('-s', '--skip'):
			config['offset'] = float(value)
		elif opt in ('-a', '--average'):
			config['average'] = float(value)
		elif opt in ('-d', '--duration'):
			config['duration'] = float(value)
		elif opt in ('-c', '--clip-level'):
			config['clip'] = int(value)
		elif opt in ('-e', '--estimate-clip'):
			config['estimate'] = True
		else:
			assert False
	
	# Add in arguments
	config['args'] = args

	# Return configuration
	return config


def bestFreqUnits(freq):
	"""Given a numpy array of frequencies in Hz, return a new array with the
	frequencies in the best units possible (kHz, MHz, etc.)."""

	# Figure out how large the data are
	scale = int(math.log10(freq.max()))
	if scale >= 9:
		divis = 1e9
		units = 'GHz'
	elif scale >= 6:
		divis = 1e6
		units = 'MHz'
	elif scale >= 3:
		divis = 1e3
		units = 'kHz'
	else:
		divis = 1
		units = 'Hz'

	# Convert the frequency
	newFreq = freq / divis

	# Return units and freq
	return (newFreq, units)


def main(args):
	#nChunks = 10000#10000, the size of a file.
	#nFramesAvg = 1*4#200, 50 frames per pol, the subintergration time.
	nChunks = 4 #the size of a file.
	nFramesAvg = 1*4 #the intergration time.
	#for offset_i in range(1500*0, 1500*1):# one offset = nChunks*nFramesAvg skiped
	#for offset_i in range(1500*2, 1500*3):# one offset = nChunks*nFramesAvg skiped
	for offset_i in range(0, 1):# one offset = nChunks*nFramesAvg skiped
		offset = nChunks*nFramesAvg*offset_i
	# Parse command line options
		config = parseOptions(args)

	# Length of the FFT
		LFFT = config['LFFT']	
	# Build the DRX file
		try:
			#drxFile = drsu.getFileByName(config['args'][0], config['args'][1])
                        fh = open(config['args'][0], "rb")
                        nFramesFile = os.path.getsize(config['args'][0]) / drx.FrameSize
		except:
			print config['args']
			sys.exit(1)

		#drxFile.open()
		#nFramesFile = drxFile.size / drx.FrameSize
	
		while True:
			try:
				junkFrame = drx.readFrame(fh)
				try:
					srate = junkFrame.getSampleRate()
					t0 = junkFrame.getTime()
					break
				except ZeroDivisionError:
					pass
			except errors.syncError:
				fh.seek(-drx.FrameSize+1, 1)
			
		fh.seek(-drx.FrameSize, 1)
	
		beam,tune,pol = junkFrame.parseID()
		beams = drx.getBeamCount(fh)
		tunepols = drx.getFramesPerObs(fh)
		tunepol = tunepols[0] + tunepols[1] + tunepols[2] + tunepols[3]
		beampols = tunepol
		config['offset'] = offset/srate/beampols*4096
		if offset != 0:
			fh.seek(offset*drx.FrameSize, 1)

	# Make sure that the file chunk size contains is an integer multiple
	# of the FFT length so that no data gets dropped.  This needs to
	# take into account the number of beampols in the data, the FFT length,
	# and the number of samples per frame.
		maxFrames = int(1.0*config['maxFrames']/beampols*4096/float(LFFT))*LFFT/4096*beampols

	# Number of frames to integrate over
#	nFramesAvg = int(config['average'] * srate / 4096 * beampols)
#	nFramesAvg = int(1.0 * nFramesAvg / beampols*4096/float(LFFT))*LFFT/4096*beampols
		config['average'] = 1.0 * nFramesAvg / beampols * 4096 / srate
		maxFrames = nFramesAvg

	# Number of remaining chunks (and the correction to the number of
	# frames to read in).
#	nChunks = int(round(config['duration'] / config['average']))
		config['duration']=nChunks*config['average']
		if nChunks == 0:
			nChunks = 1
		nFrames = nFramesAvg*nChunks
	
	# Date & Central Frequnecy
		beginDate = ephem.Date(unix_to_utcjd(junkFrame.getTime()) - DJD_OFFSET)
		centralFreq1 = 0.0
		centralFreq2 = 0.0
		for i in xrange(4):
			junkFrame = drx.readFrame(fh)
			b,t,p = junkFrame.parseID()
			if p == 0 and t == 1:
				try:
					centralFreq1 = junkFrame.getCentralFreq()
				except AttributeError:
					from lsl.common.dp import fS
					centralFreq1 = fS * ((junkFrame.data.flags>>32) & (2**32-1)) / 2**32
			elif p == 0 and t == 2:
				try:
					centralFreq2 = junkFrame.getCentralFreq()
				except AttributeError:
					from lsl.common.dp import fS
					centralFreq2 = fS * ((junkFrame.data.flags>>32) & (2**32-1)) / 2**32
			else:
				pass
		fh.seek(-4*drx.FrameSize, 1)
	
		config['freq1'] = centralFreq1
		config['freq2'] = centralFreq2

	# File summary
		print "Filename: %s" % config['args']
		print "Date of First Frame: %s" % str(beginDate)
		print "Beams: %i" % beams
		print "Tune/Pols: %i %i %i %i" % tunepols
		print "Sample Rate: %i Hz" % srate
		print "Tuning Frequency: %.3f Hz (1); %.3f Hz (2)" % (centralFreq1, centralFreq2)
		print "Frames: %i (%.3f s)" % (nFramesFile, 1.0 * nFramesFile / beampols * 4096 / srate)
		print "---"
		print "Offset: %.3f s (%i frames)" % (config['offset'], offset)
		print "Integration: %.3f s (%i frames; %i frames per beam/tune/pol)" % (config['average'], nFramesAvg, nFramesAvg / beampols)
		print "Duration: %.3f s (%i frames; %i frames per beam/tune/pol)" % (config['average']*nChunks, nFrames, nFrames / beampols)
		print "Chunks: %i" % nChunks

if __name__ == "__main__":
	main(sys.argv[1:])
