import numpy as np
# import matplotlib.pyplot as plt
import time
import os
import sys
from scipy.optimize import minimize
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from PMTgiom import make_pmts

angs=np.array([0, 41, 45, 47, 52, 60, 69, 76, 79, 90, 101, 104, 111, 120, 128, 133, 135, 139, 180])
pmts=np.array([0,1,4,6,7,3,10,13,15,17,18,5,11,12,14])
UpInd=[0,1,6]
DnInd=[4,9,10]
mid, right, up=make_pmts(pmts)
ang=[]
K=np.zeros((len(pmts), len(pmts))).astype(int)
for i in range(len(pmts)):
    for j in range(len(pmts)):
        ang.append(np.arccos(np.sum(mid[i]*mid[j]))/np.pi*180)
        K[i,j]=np.argmin(np.abs(np.arccos(np.sum(mid[i]*mid[j]))/np.pi*180-angs))
h, bins=np.histogram(ang, bins=np.arange(182)-0.5)
U=h[h>0]/np.sum(h[h>0])


blw_cut=100
init_cut=100
chi2_cut=8000
height_cut=3500
G=np.zeros((20,1000))
H=np.zeros((20,1000,len(pmts)))
spectra=np.zeros((len(pmts), 10))
spectrum=np.zeros(50)
Chi2=np.zeros(100)
BLW=np.zeros(100)
Chi2bins=np.linspace(0,15000,101)
BLWbins=np.linspace(0,500,101)
TP=np.zeros((100,50, len(pmts)))
P=np.zeros((50, len(pmts)))
Pbins=np.linspace(0.001,20,51)
Tbins=np.linspace(0,999,101)

path='/storage/xenon/gerak/301120B/PulserC/EventRecon/'
# path='/storage/xenon/gerak/011220/NG{}/EventRecon/'.format(n)
for filename in os.listdir(path):
    if filename.endswith(".npz") and filename.startswith("recon1ns"):
        print(path, filename)
        data=np.load(path+filename)
        r=data['rec']
        r=r[r['init_event']>init_cut]
        BLW+=np.histogram(np.sqrt(np.sum(r['blw']**2, axis=1)), bins=BLWbins)[0]
        r=r[np.sqrt(np.sum(r['blw']**2, axis=1))<blw_cut]
        r=r[np.sum(np.sum(r['h'], axis=1), axis=1)>0]
        r=r[np.all(r['height']<height_cut, axis=1)]
        for i in range(len(pmts)):
            P[:,i]+=np.histogram(r['P'][:,:,i], bins=Pbins)[0]
            for j in np.arange(len(Tbins[:-1])):
                Ind=np.nonzero(np.logical_and(np.arange(1000)>=Tbins[j], np.arange(1000)<Tbins[j+1]))[0]
                TP[j,:,i]+=np.histogram(r['P'][:,Ind,i], bins=Pbins)[0]


        Chi2+=np.histogram(np.sqrt(np.sum(r['chi2']**2, axis=1)), bins=Chi2bins)[0]
        r=r[np.sqrt(np.sum(r['chi2']**2, axis=1))<chi2_cut]


        h, Sbins=np.histogram(np.sum(np.sum(r['h'], axis=1), axis=1), bins=np.arange(len(spectrum)+1))
        spectrum+=h


        for i in range(len(pmts)):
            h, sbins=np.histogram(np.sum(r['h'][:,:,i], axis=1), bins=np.arange(len(spectra[i])+1))
            spectra[i]+=h

        h=r['h']
        # for i in range(np.shape(h)[0]):
        #     for j in range(len(pmts)):
        #         h[i,:,j]=np.roll(h[i,:,j], -np.amin(np.nonzero(np.sum(h, axis=2)[i]>0)[0]))

        for j in range(1000):
            G[:,j]+=np.histogram(np.sum(h[:,j,:], axis=1), bins=np.arange(np.shape(G)[0]+1))[0]
            for i in range(len(pmts)):
                H[:,j,i]+=np.histogram(h[:,j,i], bins=np.arange(np.shape(H)[0]+1))[0]

np.savez(path+'h', H=H, G=G, spectra=spectra.T, spectrum=spectrum, Sbins=Sbins, sbins=sbins,
         Chi2=Chi2, Chi2bins=Chi2bins, BLW=BLW, BLWbins=BLWbins, blw_cut=blw_cut, chi2_cut=chi2_cut, TP=TP, Pbins=Pbins, Tbins=Tbins, P=P)
