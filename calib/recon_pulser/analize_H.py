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

pmts=np.array([0,1,4,6,7,3,10,13,15,17,18,5,11,12,14])


path='/home/gerak/Desktop/DireXeno/011220/'
data=np.load('h.npz')
H=data['H']
G=data['G']
BLW=data['BLW']
BLWbins=data['BLWbins']
Chi2=data['Chi2']
Chi2bins=data['Chi2bins']
chi2_cut=data['chi2_cut']
blw_cut=data['blw_cut']
TP=data['TP']
Pbins=data['Pbins']
Tbins=data['Tbins']
Sbins=data['Sbins']
spectrum=data['spectrum']
P=data['P']


plt.figure()
plt.title('Chi2')
plt.step(0.5*(Chi2bins[1:]+Chi2bins[:-1]), Chi2, where='mid')
plt.axvline(chi2_cut, 0,1, color='k', linewidth=3)


plt.figure()
plt.title('BLW')
plt.step(0.5*(BLWbins[1:]+BLWbins[:-1]), BLW, where='mid')
plt.axvline(blw_cut, 0,1, color='k', linewidth=3)


fig, ax=plt.subplots(3,5)
X, Y= np.meshgrid(0.5*(Tbins[1:]+Tbins[:-1]), 0.5*(Pbins[1:]+Pbins[:-1]))
for i in range(len(pmts)):
    np.ravel(ax)[i].pcolor(X, Y, TP[:,:,i].T, norm=mcolors.PowerNorm(0.3))


plt.figure()
plt.title('spectrum')
plt.step(0.5*(Sbins[1:]+Sbins[1:]), spectrum, where='mid')


t=np.arange(1000)
plt.figure()
N=np.sum(np.sum(G.T*np.arange(np.shape(G)[0]), axis=1)/np.sum(G, axis=0))
plt.step(t, (np.sum(G.T*np.arange(np.shape(G)[0]), axis=1)/np.sum(G, axis=0))/N, where='mid', linewidth=3)
plt.yscale('log')


fig, ax=plt.subplots(3,5)
for i in range(15):
    np.ravel(ax)[i].step(t, np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1)/np.sum(H[:,:,i], axis=0), where='mid', label='PMT{}'.format(pmts[i]))
    np.ravel(ax)[i].set_yscale('log')
    np.ravel(ax)[i].legend()

fig, ax=plt.subplots(3,5)
fig.suptitle('P')
for i in range(15):
    np.ravel(ax)[i].step(0.5*(Pbins[1:]+Pbins[:-1]), P[:,i], where='mid')
    np.ravel(ax)[i].set_yscale('log')
plt.show()
