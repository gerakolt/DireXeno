import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
import multiprocessing
# from Sim_show import Sim_fit
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit


pmts=np.array([0,1,2,4,6,7,3,10,13,15,17,18,5,11,12,14])

# def make_time(time):
#     t=np.zeros(len(time))
#     n=0
#     t[0]=8*time[0]
#     for i in range(1,len(t)):
#         if time[i]<time[i-1] or t[i-1]+8*time[i]>(n+1)*9e9:
#             n+=1
#         t[i]=9e9*n+8*time[i]
#
#     t=t/1e9
#
#     def func(x, a,b):
#         return a*x+b
#     p0=[(t[-1]-t[0])/len(t), t[0]]
#     p, cov=curve_fit(func, np.arange(len(t)), t, p0=p0)
#
#     return t, p

# path='/home/gerak/Desktop/DireXeno/011220/EventRecon/'
path=''
data=np.load(path+'NG.npz')
trig=data['trig']
H=data['H']
G=data['G']
FSbins=data['FSbins']
FullSpectrum=data['Fullspectrum']
W=data['W']
Wbins=data['Wbins']
Wbands=data['Wbands']
Sbands=data['Sbands']
BLW=data['BLW']
BLWbins=data['BLWbins']
Chi2=data['Chi2']
Chi2bins=data['Chi2bins']
blw_cut=data['blw_cut']
chi2_cut=data['chi2_cut']
UpDn=data['UpDn']
Upbins=data['Upbins']
Dnbins=data['Dnbins']
time=data['time']

data=np.load(path+'BG.npz')
FullSpectrumBG=data['Fullspectrum']
WBG=data['W']
timeBG=data['time']


# time, p=make_time(time)
# timeBG, q=make_time(timeBG)
#
# plt.figure()
# plt.title('Over all time is {} min'.format((len(time)*p[0]+len(timeBG)*q[0])/60))
# plt.plot(np.arange(len(time)), time, 'r*')
# plt.plot(np.arange(len(time)), np.arange(len(time))*p[0]+p[1], 'r.-', label='{} Hz'.format(np.round(1/p[0], decimals=3)))
# plt.plot(np.arange(len(timeBG)), timeBG, 'b*')
# plt.plot(np.arange(len(timeBG)), np.arange(len(timeBG))*q[0]+q[1], 'b.-', label='{} Hz'.format(np.round(1/q[0], decimals=3)))
# plt.legend()

plt.figure()
plt.title('BLW')
plt.step(0.5*(BLWbins[1:]+BLWbins[:-1]), BLW, where='mid')
plt.axvline(blw_cut, 0,1, color='k', linewidth=3)

plt.figure()
plt.title('Chi2')
plt.step(0.5*(Chi2bins[1:]+BLWbins[:-1]), Chi2, where='mid')
plt.axvline(chi2_cut, 0,1, color='k', linewidth=3)


plt.figure()
plt.title('W')
plt.bar(0.5*(Wbins[1:]+Wbins[:-1]), W, width=Wbins[1:]-Wbins[:-1], color='r', alpha=0.5)
plt.bar(0.5*(Wbins[1:]+Wbins[:-1]), WBG, width=Wbins[1:]-Wbins[:-1], color='g', alpha=0.5)

for i in range(len(Wbands)):
    plt.axvline(Wbands[i], 0, 1)
plt.yscale('log')

plt.figure()
plt.title('Full Spectrum')
plt.step(0.5*(FSbins[1:]+FSbins[:-1]), np.sum(FullSpectrum, axis=0), where='mid')
plt.step(0.5*(FSbins[1:]+FSbins[:-1]), np.sum(FullSpectrumBG, axis=0), where='mid', linewidth=3, label='BG')
for i in range(len(Wbands)-1):
    plt.bar(0.5*(FSbins[1:]+FSbins[:-1]), FullSpectrum[i], width=FSbins[1:]-FSbins[:-1], alpha=0.5)
plt.yscale('log')
plt.legend()

plt.figure()
X, Y= np.meshgrid(0.5*(Upbins[1:]+Upbins[:-1]), 0.5*(Dnbins[1:]+Dnbins[:-1]))
# plt.plot(np.arange(350), np.arange(350)+55, 'k--', linewidth=5, label=r'$N_{bot}<N_{top}+55$'+'\ncut')
plt.pcolor(X, Y, UpDn.T, norm=mcolors.PowerNorm(0.3))


t=np.arange(100)
plt.figure()
for k in range(len(Sbands)-1):
    data=np.ravel(G[k])
    N=np.sum(np.sum(G[k].T*np.arange(np.shape(G)[1]), axis=1)/np.sum(G[k], axis=0))
    plt.step(t, (np.sum(G[k].T*np.arange(np.shape(G)[1]), axis=1)/np.sum(G[k], axis=0))/N, where='mid', label='PEs: {}-{}'.format(Sbands[k], Sbands[k+1]), linewidth=3)
    plt.yscale('log')

fig, ax=plt.subplots(2,3)
for k in range(len(Sbands)-1):
    #plt.title('The temporal structure in different energy ranges (NRs) {}-{}'.format(Sbands[k], Sbands[k+1]), fontsize=35)
    data=np.ravel(G[k])
    N=np.sum(np.sum(G[k].T*np.arange(np.shape(G)[1]), axis=1)/np.sum(G[k], axis=0))
    np.ravel(ax)[k].step(t, (np.sum(G[k].T*np.arange(np.shape(G)[1]), axis=1)/np.sum(G[k], axis=0))/N, where='mid', label='PEs: {}-{}'.format(Sbands[k], Sbands[k+1]), linewidth=3)
    np.ravel(ax)[k].legend(fontsize=10)
    np.ravel(ax)[k].set_yscale('log')
    # plt.xlabel('Time [ns]', fontsize=35)
    # plt.ylabel('The probability to resolve a PE /\naveage number of PEs at\n the energy range', fontsize=35)
    # plt.ylabel('The average number of\nPEs resolved (normalized)', fontsize=35)
    # plt.tick_params(axis='both', which='major', labelsize=20)

for k in range(1):
    fig, ax=plt.subplots(4,4)
    fig.suptitle('PMT temporal ({}-{})'.format(Sbands[k], Sbands[k+1]))
    for i in range(16):
        np.ravel(ax)[i].step(t, np.sum(H[k, :,:,i].T*np.arange(np.shape(H)[1]), axis=1)/np.sum(H[k, :,:,i], axis=0), where='mid', label='PMT{}'.format(pmts[i]))
        # np.ravel(ax)[i].errorbar(t, np.mean(np.sum(np.transpose(SH[k,:,:,:,i], (0,2,1))*np.arange(np.shape(H)[1]), axis=-1)/np.sum(SH[k,:,:,:,i], axis=1), axis=0),
        #             np.std(np.sum(np.transpose(SH[k,:,:,:,i], (0,2,1))*np.arange(np.shape(H)[1]), axis=-1)/np.sum(SH[k,:,:,:,i], axis=1), axis=0), fmt='.')
        np.ravel(ax)[i].set_yscale('log')
        np.ravel(ax)[i].legend()



plt.show()
