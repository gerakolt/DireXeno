from admin import make_glob_array
import numpy as np
import time
import os
import sys
from scipy.stats import poisson, binom
from scipy.special import erf as erf
from Sim import Sim_fit
import multiprocessing
# import matplotlib.pyplot as plt


pmts=[0,1,4,6,7,3,10,13,15,17,18,5,11,12,14]
Sa=np.ones(16)*2
# St=[0.7607137 ,  0.70664638,
#   0.70108794,  0.7081122,   0.88173613,  0.71983791,  0.74337253,  0.6421211,
#   0.74077786 , 0.74203365,  0.75295714 , 0.69932017,  0.71032887,  0.82576841,
#   0.78990368 , 0.68267526]
# St=np.zeros(16)+0.35

try:
    path='/home/gerak/Desktop/DireXeno/011220/'
    data=np.load(path+'H.npz')
except:
    path='/storage/xenon/gerak/011220/'
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
data=np.loadtxt(path+'NRel.txt')
NDep=[]
NSpec=[]
for i in range(len(data)):
    NDep.append(data[i][0])
    NSpec.append(data[i][1])
NSpec=np.array(NSpec/np.sum(NSpec))
NDep=np.array(NDep)


keVbins=np.linspace(0,10, 101)
PEbins=np.linspace(0,200, 101)
adn=0
bdn=0
aup=1/5
bup=10

T0=time.time()
def L(p, pmt, q, ps, ls, qs, ID, step, Ravel_KeVPE, Ravel_Spectrum, Ravel_Spectra, Ravel_G, Ravel_H, Ravel_Angs, Ravel_Angs10, Ravel_Fullspectrum,
        Ravel_W):
    # global T0
    if np.any(p<0):
        print('p<0:', np.nonzero(p<0)[0], p[np.nonzero(p<0)[0]])
        return 1e20*(1-np.amin(p))
    Q, St, W, fano, nLXe, sigma_smr, mu, R, a, F, Tf, Ts=make_glob_array(p, len(pmts), len(Sbands)-1)
    if np.any(np.array(Q)[:]>1):
        print('Q>1')
        return 1e20*np.amax(Q)
    if np.any(R>1):
        print('R>1')
        return 1e20*np.amax(R)
    if np.any(F>1):
        print('F>1')
        return 1e20*np.amax(F)
    if sigma_smr>2*np.pi:
        return 1e20*sigma_smr


    qa=multiprocessing.Array("d", [0, 0, 0, 0, 0])
    ravel_Spectrum=multiprocessing.Array("d", np.zeros(len(np.ravel(Spectrum))))
    ravel_Spectra=multiprocessing.Array("d", np.zeros(len(np.ravel(Spectra))))
    ravel_G=multiprocessing.Array("d", np.zeros(len(np.ravel(G))))
    ravel_H=multiprocessing.Array("d", np.zeros(len(np.ravel(H))))
    ravel_Angs=multiprocessing.Array("d", np.zeros(len(np.ravel(Angs))))
    ravel_Angs10=multiprocessing.Array("d", np.zeros(len(np.ravel(Angs))))
    ravel_Fullspectrum=multiprocessing.Array("d", np.zeros(len(FSbins)-1))
    ravel_W=multiprocessing.Array("d", np.zeros(len(Wbins)-1))
    KeVPE=multiprocessing.Array("d", (len(keVbins)-1)*(len(PEbins)-1))
    A=multiprocessing.Process(target=make_l, args=(len(ls)+1, qa, KeVPE, ravel_Spectrum, ravel_Spectra, ravel_G, ravel_H, ravel_Angs, ravel_Angs10,
                            ravel_Fullspectrum, ravel_W, G, H, Spectrum,
                            Spectra, Angs, w, Sbins, sbins, FSbins, Angsbins, Wbins, keVbins, PEbins,
                             'A', NDep, NSpec, Q, Sa, St, W, fano, nLXe, sigma_smr, mu, R, a, F, Tf, Ts))

    A.start()
    A.join()
    div=[1,1,1,1,1]
    # l=qa[0]/div[0]+qa[1]/div[1]+qa[2]/div[2]+qa[3]/div[3]+qa[4]/div[4]
    l=qa[2]
    Ravel_Spectrum[len(ls)%np.shape(Ravel_Spectrum)[0],0]=ravel_Spectrum
    Ravel_Spectra[len(ls)%np.shape(Ravel_Spectrum)[0],0]=ravel_Spectra
    Ravel_G[len(ls)%np.shape(Ravel_Spectrum)[0],0]=ravel_G
    Ravel_H[len(ls)%np.shape(Ravel_Spectrum)[0],0]=ravel_H
    Ravel_Angs[len(ls)%np.shape(Ravel_Spectrum)[0],0]=ravel_Angs
    Ravel_Angs10[len(ls)%np.shape(Ravel_Spectrum)[0],0]=ravel_Angs10
    Ravel_Fullspectrum[len(ls)%np.shape(Ravel_Spectrum)[0],0]=ravel_Fullspectrum
    Ravel_W[len(ls)%np.shape(Ravel_Spectrum)[0],0]=ravel_W
    Ravel_KeVPE[len(ls)%np.shape(Ravel_Spectrum)[0],0]=KeVPE

    ps[len(ls)]=p
    qs[len(ls),:,0]=np.array(qa)
    ls.append(l)
    print(len(ls), l, (time.time()-T0)/len(ls))
    np.savez('Q'.format(pmt, ID), param=q, ps=ps, ls=ls, qs=qs, step=step, Ravel_G=Ravel_G, Ravel_H=Ravel_H, Ravel_Spectrum=Ravel_Spectrum,
                Ravel_Spectra=Ravel_Spectra, Ravel_Angs=Ravel_Angs, Ravel_Angs10=Ravel_Angs10, Ravel_Fullspectrum=Ravel_Fullspectrum,
                 Ravel_W=Ravel_W, Ravel_KeVPE=Ravel_KeVPE, keVbins=keVbins, PEbins=PEbins, aup=aup, adn=adn, bup=bup, bdn=bdn, T=time.time()-T0)
    return l



def make_l(i, q, KeVPE, ravel_Spectrum, ravel_Spectra, ravel_G, ravel_H, ravel_Angs, ravel_Angs10, ravel_Fullspectrum, ravel_w, G, H, Spectrum, Spectra, Angs, w,
            Bins, bins, FSbins, Angsbins, Wbins, keVbins, PEbins, type,
            NDep, NSpec, Q, Sa, St, W, fano, nLXe, sigma_smr, mu, R, a, F, Tf, Ts):
    np.random.seed(int(i*time.time()%2**32))
    spectrum, spectra, g, h, angs, angs10, fullspectrum, SW, keVPE=Sim_fit(type, NDep, NSpec, St, Sa, Q, W,
                                                                            fano, nLXe, sigma_smr, mu, R, a, F, Tf, Ts,
                                                                            Sbins, sbins, Angsbins, np.shape(H)[1], np.shape(G)[1], FSbins, Wbins,
                                                                            Sbands, keVbins, PEbins, adn, bdn, aup, bup)
    if np.any(spectrum<0):
        q[0]=1e20*(1-np.amin(spectrum))
    else:
        for k in range(len(W)):
            rng=np.nonzero(np.logical_and(0.5*(FSbins[1:]+FSbins[:-1])>=Sbands[k], 0.5*(FSbins[1:]+FSbins[:-1])<=Sbands[k+1]))[0]
            #N=np.sum(np.sum(np.sum(FullSpectrum, axis=0)[rng]))/np.sum(fullspectrum[k, rng])
            # N=np.sum(Spectra[:,:,k])/np.sum(spectra[k])
            N=np.amax(np.sum(Spectra[:,:,k], axis=0))
            if np.sum(fullspectrum[k])>0:
                ravel_Fullspectrum[:]+=N*fullspectrum[k]/np.sum(fullspectrum[k])
            ravel_w[:]+=N*SW[k]
            spectrum[k]=N*spectrum[k]


        # ravel_w[:]+=int(np.sum(w[0.5*(Wbins[1:]+Wbins[:-1])>=0.5]))*np.array(ravel_w[:])/np.sum(np.array(ravel_w)[0.5*(Wbins[1:]+Wbins[:-1])>=0.5])
        KeVPE[:]=np.ravel(keVPE)
        ravel_w[:]+=int(np.sum(w))*np.array(ravel_w[:])/np.sum(np.array(ravel_w))

        data=Spectrum
        model=np.sum(spectrum, axis=0)
        ravel_Spectrum[:]=model
        q[0]=-np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

        data=np.ravel(np.transpose(Spectra, (2,0,1)))
        model=np.ravel(np.transpose(np.sum(Spectra, axis=0)*np.transpose(spectra, (1,2,0)), (2,0,1)))
        ravel_Spectra[:]=model
        q[1]=-np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

        data=np.ravel(G)
        model=np.ravel(np.transpose(np.sum(G, axis=1)*np.transpose(g, (1,0,2)), (1,0,2)))
        ravel_G[:]=model
        q[2]=-np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

        data=np.ravel(H)
        model=np.ravel(np.transpose(np.sum(H, axis=1)*np.transpose(h, (1,0,2,3)), (1,0,2,3)))
        ravel_H[:]=model
        q[3]=-np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

        data=np.ravel(Angs)
        model=np.ravel((np.sum(Angs, axis=1)*angs.T).T)
        ravel_Angs[:]=model
        # q[4]=-np.sum(data*np.log((model+1e-10)/(data+1e-10))+data-model)

        ravel_Angs10[:]=np.ravel((np.sum(Angs, axis=1)*angs10.T).T)


def make_Ravel():
    Ravel_Spectrum=np.zeros((10, 1, len(np.ravel(Spectrum))))
    Ravel_Spectra=np.zeros((10, 1, len(np.ravel(Spectra))))
    Ravel_G=np.zeros((10, 1, len(np.ravel(G))))
    Ravel_H=np.zeros((10, 1, len(np.ravel(H))))
    Ravel_Angs=np.zeros((10, 1, len(np.ravel(Angs))))
    Ravel_Angs10=np.zeros((10, 1, len(np.ravel(Angs))))
    Ravel_Fullspectrum=np.zeros((10, 1, len(FSbins)-1))
    Ravel_W=np.zeros((10, 1, len(np.ravel(w))))
    Ravel_KeVPE=np.zeros((10, 1, (len(keVbins)-1)*(len(PEbins)-1)))
    return Ravel_Spectrum, Ravel_Spectra, Ravel_G, Ravel_H, Ravel_Angs, Ravel_Angs10, Ravel_Fullspectrum, Ravel_W, Ravel_KeVPE
