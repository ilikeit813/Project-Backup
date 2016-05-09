import numpy as np
import matplotlib.pyplot as plt
import sys

a = np.load('jamie.npy')
b = np.load('057139_000656029_1_fft_offset_000000000_frames.npy')

print a.shape
print b.shape

#sys.exit()


#i=0

for i in range(0,4):
	plt.subplot(2,2,i+1)
	plt.plot(np.fft.fftshift(np.log10(a[:,i,:].mean(0))))
plt.savefig('jamie1')
plt.clf()


for i in range(0,4):
	plt.subplot(2,2,i+1)
	plt.plot(np.log10(b[:,i,:].mean(0)))
plt.savefig('jamie2')

