import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from admin import make_glob_array
import multiprocessing
# from Sim_show import Sim_fit
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
from make_3D import make_3D

pmts=np.array([0,1,4,6,7,3,10,13,15,17,18,5,11,12,14])


path='/home/gerak/Desktop/DireXeno/011220/'
data=np.load(path+'h.npz')
Angs=data['Angs']
Angs10=data['Angs10']
Angsbins=data['Angsbins']
H=data['H']
G=data['G']
Spectrum=data['spectrum']
Sbins=data['Sbins']
Spectra=np.transpose(data['spectra'], (2,0,1))
sbins=data['sbins']
FSbins=data['FSbins']
FullSpectrum=data['Fullspectrum']
w=data['W']
Wbins=data['Wbins']
Wbands=data['Wbands']
Sbands=data['Sbands']
W20_40=data['W20_40']
W20_40bins=data['W20_40bins']

data=np.loadtxt(path+'NRel.txt')
NDep=[]
NSpec=[]
for i in range(len(data)):
    NDep.append(data[i][0])
    NSpec.append(data[i][1])
NSpec=NSpec/np.sum(NSpec)


N=10
data=np.load('Q.npz')
ls=data['ls']
Sspectrum=data['Ravel_Spectrum'].reshape((N, 1, np.shape(Spectrum)[0]))[:,0]
Sspectra=data['Ravel_Spectra'].reshape((N, 1, np.shape(Spectra)[0], np.shape(Spectra)[1], np.shape(Spectra)[2]))[:,0]
SG=data['Ravel_G'].reshape((N, 1, np.shape(G)[0], np.shape(G)[1], np.shape(G)[2]))[:,0]
SH=data['Ravel_H'].reshape((N, 1, np.shape(H)[0], np.shape(H)[1], np.shape(H)[2], np.shape(H)[3]))[:,0]
SAngs=data['Ravel_Angs'].reshape((N, 1, np.shape(Angs)[0], np.shape(Angs)[1]))[:,0]
SFullspectrum=data['Ravel_Fullspectrum'].reshape((N, 1, np.shape(FullSpectrum)[1]))[:,0]
SW=data['Ravel_W'].reshape((N, 1, np.shape(w)[0]))[:,0]
keVbins=data['keVbins']
PEbins=data['PEbins']
keVPE=data['Ravel_KeVPE'].reshape((N, len(keVbins)-1, len(PEbins)-1))
y=np.arange(0, 600)
y1=data['adn']*y+data['bdn']
y2=data['aup']*y+data['bup']


plt.figure()
plt.title('Energy spectrum')
plt.step(NDep, NSpec, where='mid')
# plt.yscale('log')


plt.figure(figsize=(20,10))
X, Y= np.meshgrid(0.5*(keVbins[1:]+keVbins[:-1]), 0.5*(PEbins[1:]+PEbins[:-1]))
plt.pcolor(Y, X, np.mean(keVPE, axis=0), norm=mcolors.PowerNorm(0.3))
plt.plot(y, y1, 'k--', label='{}x+{}'.format(np.round(data['adn'],decimals=2), np.round(data['bdn'],decimals=2)), linewidth=5)
plt.plot(y, y2, 'k--', label='{}x+{}'.format(np.round(data['aup'],decimals=2), np.round(data['bup'],decimals=2)), linewidth=5)
plt.xlabel('PEs', fontsize=25)
plt.ylabel('keV', fontsize=25)
plt.tick_params(axis='both', which='major', labelsize=20)
cbar=plt.colorbar()
cbar.ax.tick_params(labelsize=25)
plt.legend(fontsize=35, loc='upper right')
plt.xlim(0, np.amax(PEbins))
plt.ylim(0, np.amax(keVbins))

plt.figure()
plt.title('W')
plt.bar(0.5*(Wbins[1:]+Wbins[:-1]), w, width=Wbins[1:]-Wbins[:-1], color='r', alpha=0.5)
plt.errorbar(0.5*(Wbins[1:]+Wbins[:-1]), np.mean(SW, axis=0), np.std(SW, axis=0), fmt='.')
for i in range(len(Wbands)):
    plt.axvline(Wbands[i], 0, 1)
plt.yscale('log')
#
plt.figure()
plt.title('Full Spectrum')
plt.step(0.5*(FSbins[1:]+FSbins[:-1]), np.sum(FullSpectrum, axis=0), where='mid')
for i in range(len(Wbands)-1):
    plt.bar(0.5*(FSbins[1:]+FSbins[:-1]), FullSpectrum[i], width=FSbins[1:]-FSbins[:-1], label='spectrum', alpha=0.5)
plt.errorbar(0.5*(FSbins[1:]+FSbins[:-1]), np.mean(SFullspectrum, axis=0), np.std(SFullspectrum, axis=0), fmt='.', label='A')
plt.yscale('log')
plt.legend()

plt.figure()
plt.title('Global Spectrum')
plt.bar(0.5*(Sbins[1:]+Sbins[:-1]), Spectrum, width=Sbins[1:]-Sbins[:-1], label='spectrum', color='r', alpha=0.5)
plt.errorbar(0.5*(Sbins[1:]+Sbins[:-1]), np.mean(Sspectrum, axis=0), np.std(Sspectrum, axis=0), fmt='.', label='A')
plt.legend()
plt.yscale('log')

t=np.arange(100)
fig, ax=plt.subplots(2,3)
for k in range(len(Sbands)-1):
    #plt.title('The temporal structure in different energy ranges (NRs) {}-{}'.format(Sbands[k], Sbands[k+1]), fontsize=35)
    data=np.ravel(G[k])
    model=np.ravel(np.mean(SG[:,k], axis=0))
    N=np.sum(np.sum(G[k].T*np.arange(np.shape(G)[1]), axis=1)/np.sum(G[k], axis=0))
    np.ravel(ax)[k].step(t, (np.sum(G[k].T*np.arange(np.shape(G)[1]), axis=1)/np.sum(G[k], axis=0))/N, where='mid', label='PEs: {}-{}'.format(Sbands[k], Sbands[k+1]), linewidth=3)
    np.ravel(ax)[k].errorbar(t, np.mean(np.sum(np.transpose(SG[:,k], (0,2,1))*np.arange(np.shape(G)[1]), axis=-1)/np.sum(SG[:,k], axis=1), axis=0)/N,
                np.std(np.sum(np.transpose(SG[:,k], (0,2,1))*np.arange(np.shape(G)[1]), axis=-1)/np.sum(SG[:,k], axis=1), axis=0)/N, fmt='.', label='{}'.format(-np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)))
    np.ravel(ax)[k].legend(fontsize=10)
    np.ravel(ax)[k].set_yscale('log')
    # plt.xlabel('Time [ns]', fontsize=35)
    # plt.ylabel('The probability to resolve a PE /\naveage number of PEs at\n the energy range', fontsize=35)
    # plt.ylabel('The average number of\nPEs resolved (normalized)', fontsize=35)
    # plt.tick_params(axis='both', which='major', labelsize=20)
# plt.show()

# for k in range(len(Sbands)-1):
#     fig, ax=plt.subplots(4,4)
#     fig.suptitle('PMT spectra ({}-{})'.format(Sbands[k], Sbands[k+1]))
#     for i in range(15):
#         np.ravel(ax)[i].step(0.5*(sbins[1:]+sbins[:-1]), Spectra[k,:,i], where='mid', label='A')
#         np.ravel(ax)[i].errorbar(0.5*(sbins[1:]+sbins[:-1]), np.mean(Sspectra[:,k,:,i], axis=0), np.std(Sspectra[:,k,:,i], axis=0), fmt='.', label='A')
#         np.ravel(ax)[i].legend()
#         np.ravel(ax)[i].set_yscale('log')


# for k in range(len(Sbands)-1):
for k in range(1):
    fig, ax=plt.subplots(4,4)
    fig.suptitle('PMT temporal ({}-{})'.format(Sbands[k], Sbands[k+1]))
    for i in range(15):
        np.ravel(ax)[i].step(t, np.sum(H[k, :,:,i].T*np.arange(np.shape(H)[1]), axis=1)/np.sum(H[k, :,:,i], axis=0), where='mid', label='PMT{}'.format(pmts[i]))
        # np.ravel(ax)[i].errorbar(t, np.mean(np.sum(np.transpose(SH[k,:,:,:,i], (0,2,1))*np.arange(np.shape(H)[1]), axis=-1)/np.sum(SH[k,:,:,:,i], axis=1), axis=0),
        #             np.std(np.sum(np.transpose(SH[k,:,:,:,i], (0,2,1))*np.arange(np.shape(H)[1]), axis=-1)/np.sum(SH[k,:,:,:,i], axis=1), axis=0), fmt='.')
        np.ravel(ax)[i].set_yscale('log')
        np.ravel(ax)[i].legend()
plt.show()
