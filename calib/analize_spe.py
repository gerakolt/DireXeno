import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors


pmt=0
path='/home/gerak/Desktop/DireXeno/Pulser/PMT{}/'.format(pmt)
data=np.load(path+'cuts.npz')
blw_cut=data['blw_cut']
height_cut=data['height_cut']
left=data['left']
right=data['right']
BL=np.load(path+'BL.npz')['BL']
# BL=np.zeros(1000)
dh3_cut=0.6

data=np.load(path+'spe.npz')
rec=data['rec']


fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6))=plt.subplots(3,2)

fig.suptitle('PMT{}'.format(pmt))

ax1.hist(rec['blw'], bins=100, range=[0,60], label='blw')
ax1.axvline(blw_cut, ymin=0, ymax=1, color='k')
ax1.set_yscale('log')
ax1.legend()

rec=rec[rec['blw']<blw_cut]

ax2.hist2d(rec['height'], rec['dh3'], bins=[100,100], range=[[0,500], [0,1.5]], norm=mcolors.PowerNorm(0.3))
ax2.set_xlabel('Height')
ax2.set_ylabel('(WF[maxi-3]-WF[maxi])/Height')

ax2.axhline(dh3_cut, xmin=0, xmax=1, color='k')
ax2.axvline(height_cut, ymin=0, ymax=1, color='k')
height_cut=82
ax2.axvline(height_cut, ymin=0, ymax=1, color='g')

rec=rec[rec['dh3']<dh3_cut]
def func(x, a,b,c):
    return a*np.exp(-0.5*(x-b)**2/c**2)
range=[-1500,10000]
h_areaF, bins, pat=ax4.hist(rec['area'], bins=100, range=range, histtype='step')
rec=rec[np.logical_and(rec['maxi']>left, rec['maxi']<right)]
print(rec[rec['height']>height_cut]['id'])
h_area, bins, pat=ax4.hist(rec[rec['height']>height_cut]['area'], bins=100, range=range, histtype='step')
ax4.set_ylim(1,np.amax(h_areaF)+1000)

up=7000
dn=1000
x=0.5*(bins[1:]+bins[:-1])
p0=[np.amax(h_area), x[np.argmax(h_area)], 0.5*x[np.argmax(h_area)]]
p, cov=curve_fit(func, x[np.logical_and(x<up, x>dn)], h_area[np.logical_and(x<up, x>dn)], p0=p0)
ax4.plot(x[np.logical_and(x<up, x>dn)], func(x[np.logical_and(x<up, x>dn)], *p), 'r--')

up=2500
dn=200
p0=[np.amax(h_areaF), bins[np.argmax(h_area)], 0.5*bins[np.argmax(h_area)]]
q, cov=curve_fit(func, x[np.logical_and(x<up, x>dn)], h_areaF[np.logical_and(x<up, x>dn)], p0=p0)
sigma=np.sqrt(p[2]**2-q[2]**2)/(p[1]-q[1])
ax4.plot(x[np.logical_and(x<up, x>dn)], func(x[np.logical_and(x<up, x>dn)], *q), 'g--', label=np.round(sigma,  decimals=3))
ax4.set_yscale('log')
ax4.legend()
spe=np.sum(rec['spe'], axis=0)
maxi=np.argmin(spe)
area=-(np.sum(spe[maxi-100:maxi+200])+np.sum(spe[maxi-50:maxi+150])+np.sum(spe[maxi-100:maxi+150])+np.sum(spe[maxi-50:maxi+200]))/4
spe=(p[1]-q[1])*spe/area

h_heights, bins, pat=ax3.hist(rec['height'], bins=100, label='height', range=[0,1500], histtype='step')
heights=0.5*(bins[1:]+bins[:-1])
ax3.axvline(height_cut, ymin=0, ymax=1)
ax3.set_yscale('log')
ax3.legend()



# ax4.axvline(area, 0 ,1, color='k')
x=np.arange(1000)/5
ax5.plot(x, spe, 'r.', label='mean SPE')
ax5.fill_between(x[maxi-100:maxi+200], y1=np.amin(spe), y2=0)
ax5.fill_between(x[maxi-50:maxi+150], y1=np.amin(spe), y2=0)
ax5.plot(x, x*0, 'k--')
ax3.axvline(-np.amin(spe), ymin=0, ymax=1, color='r')
ax5.legend()

ax6.hist2d(rec['height'], rec['area'], bins=[100,100], range=[[0,500], [-1000,10000]], norm=mcolors.PowerNorm(0.3))


#
# np.savez(path+'SPE', spe=spe, sigma=sigma)
# np.savez(path+'cuts', blw_cut=blw_cut, height_cut=height_cut, left=left, right=right, dh3_cut=dh3_cut)

plt.show()
