import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from admin import make_glob_array
import multiprocessing
from minimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from make_3D import make_3D


path='/home/gerak/Desktop/DireXeno/011220/'
data=np.load(path+'H.npz')
Angs=data['Angs']
Angs10=data['Angs10']
Angsbins=data['Angsbins']
H=data['H']
G=data['G']
Spectrum=data['spectrum']
Sbins=data['Sbins']
Spectra=data['spectra']
sbins=data['sbins']
FSbins=data['FSbins']
FullSpectrum=data['Fullspectrum']
w=data['W']
Wbins=data['Wbins']
Sbands=data['Sbands']
minimize(-1, 0, 0)

N=10
data=np.load('Q2.npz')
ls=data['ls']
Sspectrum=data['Ravel_Spectrum'].reshape((N, 1, np.shape(Spectrum)[0]))[:,0]
Sspectra=data['Ravel_Spectra'].reshape((N, 1, np.shape(Spectra)[0], np.shape(Spectra)[1], np.shape(Spectra)[2]))[:,0]
SG=data['Ravel_G'].reshape((N, 5, np.shape(G)[0], np.shape(G)[1], np.shape(G)[2]))
SH=data['Ravel_H'].reshape((N, 5, np.shape(H)[0], np.shape(H)[1], np.shape(H)[2], np.shape(H)[3]))
SAngs=data['Ravel_Angs'].reshape((N, 1, np.shape(Angs)[0], np.shape(Angs)[1]))[:,0]
SFullspectrum=data['Ravel_Fullspectrum'].reshape((N, 1, np.shape(FullSpectrum)[1]))[:,0]
SW=data['Ravel_W'].reshape((N, 1, np.shape(w)[0]))[:,0]



plt.figure()
plt.title('Global Spectrum')
plt.step(0.5*(Sbins[1:]+Sbins[:-1]), Spectrum, where='mid', label='spectrum')
plt.errorbar(0.5*(Sbins[1:]+Sbins[:-1]), np.mean(Sspectrum, axis=0), np.std(Sspectrum, axis=0), fmt='.', label='A')
plt.legend()
plt.yscale('log')
t=np.arange(100)

plt.figure()
for j in range(len(Sbands)-1):
    N=np.sum(np.sum(G[j].T*np.arange(np.shape(G)[1]), axis=1)/np.sum(G[j], axis=0))
    plt.step(t, (np.sum(G[j].T*np.arange(np.shape(G)[1]), axis=1)/np.sum(G[j], axis=0))/N, where='mid', label='{}-{}'.format(Sbands[j], Sbands[j+1]))
plt.yscale('log')
plt.legend()

fig, ax=plt.subplots(3,3)
fig.suptitle('Global temporal')
for j in range(len(Sbands)-1):
    np.ravel(ax)[j].step(t, (np.sum(G[j].T*np.arange(np.shape(G)[1]), axis=1)/np.sum(G[j], axis=0)), where='mid', label='A', linewidth=6)
    np.ravel(ax)[j].errorbar(t, np.mean(np.sum(np.transpose(SG[:,0,j], (0,2,1))*np.arange(np.shape(G)[1]), axis=-1)/np.sum(SG[:,0,j], axis=1), axis=0),
                np.std(np.sum(np.transpose(SG[:,0,j], (0,2,1))*np.arange(np.shape(G)[1]), axis=-1)/np.sum(SG[:,0,j], axis=1), axis=0), fmt='.', label='A')

    lbl=['recomb fast', 'recomb slow', 'fast', 'slow']
    for k in range(1,5):
        np.ravel(ax)[j].plot(t, np.mean(np.sum(np.transpose(SG[:,k,j], (0,2,1))*np.arange(np.shape(G)[1]), axis=-1)/np.sum(SG[:,k,j], axis=1), axis=0),
         '.' ,label=lbl[k-1])
    plt.legend()
    np.ravel(ax)[j].set_yscale('log')

R=[0.24036, 0.24232, 0.25, 0.26719, 0.4574, 0.54, 0.6, 0.73311, 0.74406]
F=[0.94, 0.9, 0.81669, 0.67115, 0.6, 0.52581, 0.40951, 0.30378, 0.31024]
PE=0.5*(Sbands[1:]+Sbands[:-1])
fig, ax=plt.subplots(1,2)
ax[0].plot(PE, R, 'o', label='R')
ax[1].plot(PE, F, 'o', label='F')
ax[0].legend()
ax[1].legend()

# for k in range(len(Sbands)-1):
#     fig, ax=plt.subplots(4,4)
#     fig.suptitle('PMT spectra')
#     for i in range(15):
#         np.ravel(ax)[i].step(0.5*(sbins[1:]+sbins[:-1]), Spectra[:,i,k], where='mid', label='A')
#         np.ravel(ax)[i].errorbar(0.5*(sbins[1:]+sbins[:-1]), np.mean(Sspectra[:,:,i,k], axis=0), np.std(Sspectra[:,:,i,k], axis=0), fmt='.', label='A')
#         np.ravel(ax)[i].legend()
# fig, ax=plt.subplots(4,4)
# fig.suptitle('PMT temporal')
# for i in range(15):
#     np.ravel(ax)[i].step(t, np.sum(H[:,:,i].T*np.arange(np.shape(H)[0]), axis=1)/np.sum(H[:,:,i], axis=0), where='mid', label='A')
#     np.ravel(ax)[i].errorbar(t, np.mean(np.sum(np.transpose(SH[:,0,:,:,i], (0,2,1))*np.arange(np.shape(H)[0]), axis=-1)/np.sum(SH[:,0,:,:,i], axis=1), axis=0),
#                 np.std(np.sum(np.transpose(SH[:,0,:,:,i], (0,2,1))*np.arange(np.shape(H)[0]), axis=-1)/np.sum(SH[:,0,:,:,i], axis=1), axis=0), fmt='.', label='A')
#     np.ravel(ax)[i].set_yscale('log')
plt.show()
