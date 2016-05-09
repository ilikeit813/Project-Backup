import numpy as np
import matplotlib.pyplot as plt
import glob

fn = sorted(glob.glob('05*.npy'))
bp0 = np.load('bandpassp0.npy')
bp1 = np.load('bandpassp1.npy')

for i in (30,456,802,1342):
	plt.plot((np.load(fn[i])[:,0,360:3700]/bp0).mean(1))
	plt.savefig('jp0ts_%.4i' % i)
	plt.clf()
	plt.plot((np.load(fn[i])[:,0,360:3700]/bp0).mean(0))
	plt.savefig('jp0bp_%.4i' % i)
	plt.clf()
	plt.plot((np.load(fn[i])[:,1,360:3700]/bp1).mean(1))
	plt.savefig('jp1ts_%.4i' % i)
	plt.clf()
	plt.plot((np.load(fn[i])[:,0,360:3700]/bp0).mean(0))
	plt.savefig('jp1bp_%.4i' % i)
	plt.clf()
