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
pmts=np.array([0,1,2,4,6,7,3,10,13,15,17,18,5,11,12,14])
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
flash_cut_up=1.1
flash_cut_dn=0.02



spectrum=np.zeros(5)
spectra=np.zeros((len(pmts), 10))
Angs=np.zeros((len(angs), 50))
Angs10=np.zeros((len(angs), 50))
Angsbins=np.zeros(51)
W=np.zeros(50)
Wbands=[0.02, 0.1, 0.34, 0.6, 1]
#Sbands=[15,30,45]
Sbands=[3,9,15,22,30,40,50]
Fullspectrum=np.zeros((len(Wbands)-1, 250))
FSbins=np.arange((np.shape(Fullspectrum)[1]+1))*3
G=np.zeros((len(Sbands)-1,20,100))
H=np.zeros((len(Sbands)-1,20,100,len(pmts)))
spectra=np.zeros((len(Sbands)-1,len(pmts), 10))
spectrum=np.zeros(5*(len(Sbands)-1))
Chi2=np.zeros(100)
BLW=np.zeros(100)
Chi2bins=np.linspace(0,15000,101)
BLWbins=np.linspace(0,500,101)
Sbins=np.linspace(np.amin(Sbands), np.amax(Sbands), np.shape(spectrum)[0]+1)
sbins=np.linspace(0,20, np.shape(spectra)[1]+1)
Wbins=np.linspace(0,1,len(W)+1)
binsX=np.arange(101)
binsY=np.arange(100)
UpDn=np.zeros((len(binsX)-1, len(binsY)-1))
trig=np.zeros(10)
time=np.array([])

for n in range(4,5):
    path='/home/gerak/Desktop/DireXeno/011220/EventRecon/'
    # path='/storage/xenon/gerak/011220/NG{}/EventRecon/'.format(n)

    for filename in os.listdir(path):
        if filename.endswith(".npz") and filename.startswith("recon1ns"):
            print(path, filename)
            data=np.load(path+filename)
            r=data['rec']
            trig+=np.histogram(r['NGtrig'], bins=10, range=(0,4500))[0]
            r=r[r['NGtrig']==1]
            time=np.concatenate((time, r['time']))
            r=r[r['init_event']>init_cut]
            BLW+=np.histogram(np.sqrt(np.sum(r['blw']**2, axis=1)), bins=BLWbins)[0]
            r=r[np.sqrt(np.sum(r['blw']**2, axis=1))<blw_cut]
            r=r[np.sum(np.sum(r['h'], axis=1), axis=1)>0]
            r=r[np.all(r['height']<height_cut, axis=1)]


            Chi2+=np.histogram(np.sqrt(np.sum(r['chi2']**2, axis=1)), bins=Chi2bins)[0]
            r=r[np.sqrt(np.sum(r['chi2']**2, axis=1))<chi2_cut]
            init=np.sum(np.sum(r['h'][:,:50], axis=1), axis=1)
            full=np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)
            h, Wbins=np.histogram(init/full, bins=Wbins)
            W+=h
            up=np.sum(np.sum(r['h'][:,:500,UpInd], axis=1), axis=1)
            dn=np.sum(np.sum(r['h'][:,:500,DnInd], axis=1), axis=1)

            h, binsX, binsY=np.histogram2d(up, dn, bins=[binsX, binsY])
            UpDn+=h


            for i in range(np.shape(Fullspectrum)[0]):
                if i==np.shape(Fullspectrum)[0]-1:
                    inds=np.nonzero(np.logical_and(init/full>=Wbands[i], init/full<=Wbands[i+1]))[0]
                else:
                    inds=np.nonzero(np.logical_and(init/full>=Wbands[i], init/full<Wbands[i+1]))[0]
                h, FSbins=np.histogram(np.sum(np.sum(r['h'][inds,:500], axis=1), axis=1), bins=FSbins)
                Fullspectrum[i]+=h
            r=r[np.logical_and(init/full>flash_cut_dn, init/full<flash_cut_up)]

            h, Sbins=np.histogram(np.sum(np.sum(r['h'][:,:500], axis=1), axis=1), bins=Sbins)
            spectrum+=h

            for k in range(len(Sbands)-1):
                if k+1==len(Sbands)-1:
                    r0=r[np.logical_and(np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)<=Sbands[k+1], np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)>=Sbands[k])]
                else:
                    r0=r[np.logical_and(np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)<Sbands[k+1], np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)>=Sbands[k])]


                for i in range(len(pmts)):
                    h, sbins=np.histogram(np.sum(r0['h'][:,:500,i], axis=1), bins=np.shape(spectra)[2], range=[0,20])
                    spectra[k,i]+=h
                for j in range(100):
                    G[k,:,j]+=np.histogram(np.sum(np.sum(r0['h'][:,5*j:5*(j+1)], axis=1), axis=1), bins=np.arange(np.shape(G)[1]+1))[0]
                    for i in range(len(pmts)):
                        H[k,:,j,i]+=np.histogram(np.sum(r0['h'][:,5*j:5*(j+1),i], axis=1), bins=np.arange(np.shape(H)[1]+1))[0]

# for n in range(1,5):
#     #path='/home/gerak/Desktop/DireXeno/011220/EventRecon/'
#     path='/storage/xenon/gerak/021220/NG{}/EventRecon/'.format(n)
#     for filename in os.listdir(path):
#         if filename.endswith(".npz") and filename.startswith("recon1ns"):
#             print(path, filename)
#             data=np.load(path+filename)
#             r=data['rec']
#             trig+=np.histogram(r['NGtrig'], bins=10, range=(0,4500))[0]
#             r=r[r['NGtrig']==0]
#             r=r[r['init_event']>init_cut]
#             BLW+=np.histogram(np.sqrt(np.sum(r['blw']**2, axis=1)), bins=BLWbins)[0]
#             r=r[np.sqrt(np.sum(r['blw']**2, axis=1))<blw_cut]
#             r=r[np.sum(np.sum(r['h'], axis=1), axis=1)>0]
#             r=r[np.all(r['height']<height_cut, axis=1)]
#
#
#             Chi2+=np.histogram(np.sqrt(np.sum(r['chi2']**2, axis=1)), bins=Chi2bins)[0]
#             r=r[np.sqrt(np.sum(r['chi2']**2, axis=1))<chi2_cut]
#             init=np.sum(np.sum(r['h'][:,:50], axis=1), axis=1)
#             full=np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)
#             h, Wbins=np.histogram(init/full, bins=Wbins)
#             W+=h
#             up=np.sum(np.sum(r['h'][:,:500,UpInd], axis=1), axis=1)
#             dn=np.sum(np.sum(r['h'][:,:500,DnInd], axis=1), axis=1)
#
#             h, binsX, binsY=np.histogram2d(up, dn, bins=[binsX, binsY])
#             UpDn+=h
#
#
#             for i in range(np.shape(Fullspectrum)[0]):
#                 if i==np.shape(Fullspectrum)[0]-1:
#                     inds=np.nonzero(np.logical_and(init/full>=Wbands[i], init/full<=Wbands[i+1]))[0]
#                 else:
#                     inds=np.nonzero(np.logical_and(init/full>=Wbands[i], init/full<Wbands[i+1]))[0]
#                 h, FSbins=np.histogram(np.sum(np.sum(r['h'][inds,:500], axis=1), axis=1), bins=FSbins)
#                 Fullspectrum[i]+=h
#             r=r[np.logical_and(init/full>flash_cut_dn, init/full<flash_cut_up)]
#
#             h, Sbins=np.histogram(np.sum(np.sum(r['h'][:,:500], axis=1), axis=1), bins=Sbins)
#             spectrum+=h
#
#             for k in range(len(Sbands)-1):
#                 if k+1==len(Sbands)-1:
#                     r0=r[np.logical_and(np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)<=Sbands[k+1], np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)>=Sbands[k])]
#                 else:
#                     r0=r[np.logical_and(np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)<Sbands[k+1], np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)>=Sbands[k])]
#
#
#                 for i in range(len(pmts)):
#                     h, sbins=np.histogram(np.sum(r0['h'][:,:500,i], axis=1), bins=np.shape(spectra)[2], range=[0,20])
#                     spectra[k,i]+=h
#                 for j in range(100):
#                     G[k,:,j]+=np.histogram(np.sum(np.sum(r0['h'][:,5*j:5*(j+1)], axis=1), axis=1), bins=np.arange(np.shape(G)[1]+1))[0]
#                     for i in range(len(pmts)):
#                         H[k,:,j,i]+=np.histogram(np.sum(r0['h'][:,5*j:5*(j+1),i], axis=1), bins=np.arange(np.shape(H)[1]+1))[0]
#
#
# for n in range(1,3):
#     #path='/home/gerak/Desktop/DireXeno/011220/EventRecon/'
#     path='/storage/xenon/gerak/031220/NG{}/EventRecon/'.format(n)
#     for filename in os.listdir(path):
#         if filename.endswith(".npz") and filename.startswith("recon1ns"):
#             print(path, filename)
#             data=np.load(path+filename)
#             r=data['rec']
#             trig+=np.histogram(r['NGtrig'], bins=10, range=(0,4500))[0]
#             r=r[r['NGtrig']==0]
#             r=r[r['init_event']>init_cut]
#             BLW+=np.histogram(np.sqrt(np.sum(r['blw']**2, axis=1)), bins=BLWbins)[0]
#             r=r[np.sqrt(np.sum(r['blw']**2, axis=1))<blw_cut]
#             r=r[np.sum(np.sum(r['h'], axis=1), axis=1)>0]
#             r=r[np.all(r['height']<height_cut, axis=1)]
#
#
#             Chi2+=np.histogram(np.sqrt(np.sum(r['chi2']**2, axis=1)), bins=Chi2bins)[0]
#             r=r[np.sqrt(np.sum(r['chi2']**2, axis=1))<chi2_cut]
#             init=np.sum(np.sum(r['h'][:,:50], axis=1), axis=1)
#             full=np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)
#             h, Wbins=np.histogram(init/full, bins=Wbins)
#             W+=h
#             up=np.sum(np.sum(r['h'][:,:500,UpInd], axis=1), axis=1)
#             dn=np.sum(np.sum(r['h'][:,:500,DnInd], axis=1), axis=1)
#
#             h, binsX, binsY=np.histogram2d(up, dn, bins=[binsX, binsY])
#             UpDn+=h
#
#
#             for i in range(np.shape(Fullspectrum)[0]):
#                 if i==np.shape(Fullspectrum)[0]-1:
#                     inds=np.nonzero(np.logical_and(init/full>=Wbands[i], init/full<=Wbands[i+1]))[0]
#                 else:
#                     inds=np.nonzero(np.logical_and(init/full>=Wbands[i], init/full<Wbands[i+1]))[0]
#                 h, FSbins=np.histogram(np.sum(np.sum(r['h'][inds,:500], axis=1), axis=1), bins=FSbins)
#                 Fullspectrum[i]+=h
#             r=r[np.logical_and(init/full>flash_cut_dn, init/full<flash_cut_up)]
#
#             h, Sbins=np.histogram(np.sum(np.sum(r['h'][:,:500], axis=1), axis=1), bins=Sbins)
#             spectrum+=h
#
#             for k in range(len(Sbands)-1):
#                 if k+1==len(Sbands)-1:
#                     r0=r[np.logical_and(np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)<=Sbands[k+1], np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)>=Sbands[k])]
#                 else:
#                     r0=r[np.logical_and(np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)<Sbands[k+1], np.sum(np.sum(r['h'][:,:500], axis=1), axis=1)>=Sbands[k])]
#
#
#                 for i in range(len(pmts)):
#                     h, sbins=np.histogram(np.sum(r0['h'][:,:500,i], axis=1), bins=np.shape(spectra)[2], range=[0,20])
#                     spectra[k,i]+=h
#                 for j in range(100):
#                     G[k,:,j]+=np.histogram(np.sum(np.sum(r0['h'][:,5*j:5*(j+1)], axis=1), axis=1), bins=np.arange(np.shape(G)[1]+1))[0]
#                     for i in range(len(pmts)):
#                         H[k,:,j,i]+=np.histogram(np.sum(r0['h'][:,5*j:5*(j+1),i], axis=1), bins=np.arange(np.shape(H)[1]+1))[0]


np.savez('BG', H=H, G=G, spectra=spectra.T, spectrum=spectrum, Sbins=Sbins, sbins=sbins, Angs=Angs, Angs10=Angs10,
        Angsbins=Angsbins, Fullspectrum=Fullspectrum, FSbins=FSbins, W=W, Wbins=Wbins, Wbands=Wbands, Sbands=Sbands,
         Chi2=Chi2, Chi2bins=Chi2bins, BLW=BLW, BLWbins=BLWbins, blw_cut=blw_cut, chi2_cut=chi2_cut, UpDn=UpDn, Upbins=binsX, Dnbins=binsY, trig=trig,
         time=time)
