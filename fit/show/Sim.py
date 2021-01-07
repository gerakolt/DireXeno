import multiprocessing
import numpy as np
import sys
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb
from scipy.signal import convolve2d
import time
from admin import make_iter
from PMTgiom import whichPMT, make_K
import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

pmts=np.array([0,1,4,6,7,3,10,13,15,17,18,5,11,12,14])
angs=np.array([0, 41, 45, 47, 52, 60, 69, 76, 79, 90, 101, 104, 111, 120, 128, 133, 135, 139, 180])
K, U=make_K(pmts, angs)

def Sim_fit(type, NDep, NSpec, St, Sa, Q, W, fano, nLXe, sigma_smr, mu, R, a, F, Tf, Ts, Bins, bins, Angsbins, H0, G0, FSbins, Wbins, Sbands):
    N_events=2000
    Fullspectrum=np.zeros((len(W), len(FSbins)-1))
    spectrum=np.zeros((len(W), len(Bins)-1))
    spectra=np.zeros((len(Sbands)-1 ,len(bins)-1, len(Q)))
    H=np.zeros((5, len(Sbands)-1, H0,100,len(Q)))
    G=np.zeros((5, len(Sbands)-1, G0, 100))
    w=np.zeros((len(W), len(Wbins)-1))
    Angs=np.ones((len(angs), len(Angsbins)-1))
    Angs10=np.ones((len(angs), len(Angsbins)-1))
    for k in range(len(W)):
        N=np.array([]).astype(int)
        # if not (k==1):
        #     continue
        while len(N)<N_events:
            NE=np.random.choice(NDep, size=N_events-len(N), p=NSpec)
            intE=np.round(1000*NE/W[k]).astype(int)
            NE=NE[intE>0]
            eps=11.5*NE*54**(-7/3)
            x=0.1394*(3*eps**0.15+0.7*eps**0.6+eps)
            L=x/(1+x)
            if np.any(intE>0):
                NR=np.random.binomial(intE[intE>0], L)
                if np.any(NR>0):
                    NR=NR[NR>0]
                    n=np.round(np.random.normal(NR, np.sqrt(NR*fano), len(NR))).astype(int)
                    N=np.append(N, n[n>0])
        v=make_v(N_events, mu, type)
        p=multiprocessing.Pool(processes=2)
        ret=p.map(make_d, make_iter(N, Q, St, Sa, nLXe, sigma_smr, R[k], a, F[k], Tf, Ts, v))
        p.close()
        p.join()
        Ret=np.array(ret)
        ind=np.nonzero(np.sum(np.sum(Ret[:,0], axis=1), axis=1)>0)[0]
        if len(ind)==0:
            continue
        Ret=Ret[ind]
        Fullspectrum[k]=np.histogram(np.sum(np.sum(Ret[:,0], axis=1), axis=1), bins=FSbins)[0]
        w[k]=np.histogram(np.sum(np.sum(Ret[:,0,:10], axis=1), axis=1)/np.sum(np.sum(Ret[:,0,:100], axis=1), axis=1), bins=Wbins)[0]
        w[k]=w[k]/np.sum(w[k])
        spectrum[k]=np.histogram(np.sum(np.sum(Ret[:,0], axis=1), axis=1), bins=Bins)[0]
        spectrum[k]=spectrum[k]/np.sum(spectrum[k])
        if k+1==len(W):
            ind=np.nonzero(np.logical_and(np.sum(np.sum(Ret[:,0], axis=1), axis=1)>=Sbands[k], np.sum(np.sum(Ret[:,0], axis=1), axis=1)<=Sbands[k+1]))[0]
        else:
            ind=np.nonzero(np.logical_and(np.sum(np.sum(Ret[:,0], axis=1), axis=1)>=Sbands[k], np.sum(np.sum(Ret[:,0], axis=1), axis=1)<Sbands[k+1]))[0]
        if len(ind)==0:
            continue
        Ret0=Ret[ind]
        Ret1=np.zeros(len(np.ravel(Ret0)))
        ind=np.nonzero(np.ravel(Ret0)>0)[0]
        Ret1[ind]=np.round(np.random.normal(np.ravel(Ret0)[ind], 0.5*np.sqrt(np.ravel(Ret0)[ind]), size=len(ind)))
        Ret1=np.reshape(Ret1, np.shape(Ret0))
        # h=np.zeros((len(angs), np.shape(Ret)[0]))
        # h10=np.zeros((len(angs), np.shape(Ret)[0]))
        for i in range(len(Q)):
            spectra[k,:,i]=np.histogram(np.sum(Ret1[:,0,:,i], axis=1), bins=bins)[0]
            spectra[k,:,i]=spectra[k,:,i]/np.sum(spectra[k,:,i])
        #     for j in range(len(Q)):
        #         h[K[i,j]]+=np.sum(Ret[:,:,i], axis=1)*np.sum(Ret[:,:,j], axis=1)
        #         h10[K[i,j]]+=np.sum(Ret[:,:10,i], axis=1)*np.sum(Ret[:,:10,j], axis=1)
        # h=(h/np.sum(h, axis=0))
        # h10=(h10/np.sum(h10, axis=0))
        # for k in range(len(Angs)):
        #     Angs[k]+=np.histogram(h[k]/U[k], bins=Angsbins)[0]
            # Angs10[k]+=np.histogram(h10[k]/U[k], bins=Angsbins)[0]
        for l in range(5):
            for i in range(np.shape(Ret1)[2]):
                G[l,k,:,i]=np.histogram(np.sum(Ret1[:,l,i,:], axis=1), bins=np.arange((G0+1)))[0]
                G[l,k,:,i]=G[l,k,:,i]/np.sum(G[l,k,:,i])
                for j in range(len(Q)):
                    H[l,k,:,i,j]=np.histogram(Ret1[:,l,i,j], bins=np.arange((H0+1)))[0]
                    H[l,k,:,i,j]=H[l,k,:,i,j]/np.sum(H[l,k,:,i,j])
    return spectrum, spectra, G, H, (Angs.T/np.sum(Angs, axis=1)).T, (Angs10.T/np.sum(Angs10, axis=1)).T, Fullspectrum, w

def make_v(N, mu, type):
    if mu>10:
        costheta=np.random.uniform(-1,1,N)
        phi=np.random.uniform(0,2*np.pi,N)
        r3=np.random.uniform(0,(10/40)**3,N)
        r=r3**(1/3)
        vs=np.array([r*np.sin(np.arccos(costheta))*np.cos(phi), r*np.sin(np.arccos(costheta))*np.sin(phi), r*costheta])
    else:
        vs=np.zeros((3, N))
        count=0
        while count<N:
            d=np.random.exponential(mu/4, N-count)
            d=d[d<0.5]
            c=len(d)
            x=d-0.25
            r=np.random.uniform(0, np.sqrt(0.25**2-x**2), c)
            phi=np.random.uniform(0,2*np.pi, c)
            y=r*np.cos(phi)
            z=r*np.sin(phi)
            if type=='A':
                vs[0, count:count+c]=x
                vs[1, count:count+c]=y
                vs[2, count:count+c]=z
            elif type=='B':
                vs[0, count:count+c]=y
                vs[1, count:count+c]=-x
                vs[2, count:count+c]=z
            elif type=='C':
                vs[0, count:count+c]=-x
                vs[1, count:count+c]=y
                vs[2, count:count+c]=z
            count+=c
    return vs

def make_d(iter):
    N, Q, St, Sa, nLXe, sigma_smr, R, alpha, F, Tf, Ts, v, i=iter
    np.random.seed(int(i*time.time()%(2**32)))
    h=np.zeros((5, 100,len(Q)))
    Strig=2
    T0=40
    trig=np.random.normal(0, Strig, 1)
    recomb=np.random.binomial(N, R)
    ex=N-recomb
    t=np.zeros(N)
    if recomb>0:
        # u=np.random.uniform(size=recomb)
        # a=alpha*recomb
        # t[:recomb]+=1/a*(u/(1-u))
        t[:recomb]+=np.random.exponential(alpha, len(t[:recomb]))
    sng_ind=np.random.choice(2, size=N, replace=True, p=[F, 1-F])
    t[sng_ind==0]+=np.random.exponential(Tf, len(t[sng_ind==0]))
    t[sng_ind==1]+=np.random.exponential(Ts, len(t[sng_ind==1]))
    costheta=np.random.uniform(-1,1, N)
    phi=np.random.uniform(0,2*np.pi, N)
    us=np.vstack((np.vstack((np.sin(np.arccos(costheta))*np.cos(phi), np.sin(np.arccos(costheta))*np.sin(phi))), costheta))
    vs=np.repeat(v, N).reshape(3, N)
    count=np.zeros(N)
    while np.any(np.sqrt(np.sum(vs**2, axis=0))<0.75):

        absorb=np.nonzero(count>50)[0]
        vs[0,absorb]=2
        us[:,absorb]=vs[:,absorb]

        ind_LXe=np.nonzero(np.sqrt(np.sum(vs**2, axis=0))<=0.25)[0]
        ind_toLXe=np.nonzero(np.logical_and(np.sqrt(np.sum(vs**2, axis=0))>0.25, np.sum(vs*us, axis=0)<=0))[0]
        ind_toVac=np.nonzero(np.logical_and(np.logical_and(np.sqrt(np.sum(vs**2, axis=0))<0.75, np.sqrt(np.sum(vs**2, axis=0))>0.25), np.sum(vs*us, axis=0)>0))[0]

        count[ind_LXe]+=1
        count[ind_toLXe]+=1
        count[ind_toVac]+=1

        if len(ind_LXe)>0:
            vs[:,ind_LXe], us[:,ind_LXe]=traceLXe(vs[:,ind_LXe], us[:,ind_LXe], nLXe, sigma_smr)
        if len(ind_toLXe)>0:
            vs[:,ind_toLXe], us[:,ind_toLXe]=tracetoLXe(vs[:,ind_toLXe], us[:,ind_toLXe], nLXe, sigma_smr)
        if len(ind_toVac)>0:
            vs[:,ind_toVac], us[:,ind_toVac]=tracetoVac(vs[:,ind_toVac], us[:,ind_toVac], nLXe, sigma_smr)
    pmt_hit=whichPMT(vs, us)

    t0=np.zeros(len(Q))+1000
    tj=[]
    PE_Extrct=[]
    for j in range(len(Q)):
        hit_ind=np.nonzero(pmt_hit==j)[0]
        PE_Extrct.append(hit_ind[np.nonzero(np.random.choice(2, size=len(hit_ind), replace=True, p=[Q[j], 1-Q[j]])==0)[0]])
        tj.append(np.random.normal(trig+T0+t[PE_Extrct[j]], St[j], len(PE_Extrct[j])))
        if len(tj[j])>0:
            t0[j]=np.amin(tj[j])
    for j in range(len(Q)):
        PE_extrct=PE_Extrct[j]
        h[0,:,j]=np.histogram(tj[j]-np.amin(t0), bins=np.arange(101))[0]
        if len(np.nonzero(np.logical_and(PE_extrct<recomb, sng_ind[PE_extrct]==0)))>0:
            h[1,:,j]=np.histogram((tj[j])[np.nonzero(np.logical_and(PE_extrct<recomb, sng_ind[PE_extrct]==0))[0]]-np.amin(t0),
             bins=np.arange(101))[0]
        if len(np.nonzero(np.logical_and(PE_extrct<recomb, sng_ind[PE_extrct]==1)))>0:
            h[2,:,j]=np.histogram((tj[j])[np.nonzero(np.logical_and(PE_extrct<recomb, sng_ind[PE_extrct]==1))[0]]-np.amin(t0),
             bins=np.arange(101))[0]
        if len(np.nonzero(np.logical_and(PE_extrct>=recomb, sng_ind[PE_extrct]==0)))>0:
            h[3,:,j]=np.histogram((tj[j])[np.nonzero(np.logical_and(PE_extrct>=recomb, sng_ind[PE_extrct]==0))[0]]-np.amin(t0),
             bins=np.arange(101))[0]
        if len(np.nonzero(np.logical_and(PE_extrct>=recomb, sng_ind[PE_extrct]==1)))>0:
            h[4,:,j]=np.histogram((tj[j])[np.nonzero(np.logical_and(PE_extrct>=recomb, sng_ind[PE_extrct]==1))[0]]-np.amin(t0),
             bins=np.arange(101))[0]
    return h



def smr(vs, sigma_smr):
    # return vs
    x=np.random.uniform(-1,1, len(vs[0]))
    y=np.random.uniform(-1,1, len(vs[0]))
    z=-(vs[0]*x+vs[1]*y)/vs[2]
    rot=np.vstack((np.vstack((x, y)), z))
    rot=rot/np.sqrt(np.sum(rot**2, axis=0))
    theta=np.random.normal(0, sigma_smr, len(x))
    return rot*np.sum(rot*vs, axis=0)+np.cos(theta)*np.cross(np.cross(rot, vs, axis=0), rot, axis=0)+np.sin(theta)*np.cross(rot, vs, axis=0)



def traceLXe(vs, us, nLXe, sigma):
    nHPFS=1.6
    # us and vs is an (3,N) array
    a=(np.sqrt(np.sum(vs*us, axis=0)**2+(0.25**2-np.sum(vs**2, axis=0)))-np.sum(us*vs, axis=0)) # N len array

    vmin=np.amin(np.sqrt(np.sum(vs**2, axis=0)))
    ind=np.argmin(np.sqrt(np.sum(vs**2, axis=0)))
    vs=vs+us*a
    rot=np.cross(us,smr(vs, sigma), axis=0)
    rot=rot/np.sqrt(np.sum(rot**2, axis=0))
    inn=np.arccos(np.sum(vs*us, axis=0)/np.sqrt(np.sum(vs**2, axis=0))) # N len array
    TIR=np.nonzero(np.sin(inn)*nLXe/nHPFS>1)[0]
    Rif=np.nonzero(np.sin(inn)*nLXe/nHPFS<=1)[0]

    if len(Rif)>0:
        out=np.arcsin(np.sin(inn[Rif])*nLXe/nHPFS) # N len array
        theta=inn[Rif]-out
        us[:,Rif]=rot[:,Rif]*np.sum(rot[:,Rif]*us[:,Rif], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,Rif], us[:,Rif], axis=0), rot[:,Rif], axis=0)+np.sin(theta)*np.cross(rot[:,Rif], us[:,Rif], axis=0)

    if len(TIR)>0:
        theta=-(np.pi-2*inn[TIR])
        us[:,TIR]=rot[:, TIR]*np.sum(rot[:,TIR]*us[:,TIR], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,TIR], us[:,TIR], axis=0), rot[:,TIR], axis=0)+np.sin(theta)*np.cross(rot[:,TIR], us[:,TIR], axis=0)

    us=us/np.sqrt(np.sum(us*us, axis=0))
    return vs+1e-6*us, us

def tracetoLXe(vs, us, nLXe, sigma):
    nHPFS=1.6
    # us and vs is an (3,N) array
    d=np.sum(vs*us, axis=0)**2+(0.25**2-np.sum(vs**2, axis=0))
    toLXe=np.nonzero(d>=0)[0]
    toHPFS=np.nonzero(d>=0)[0]
    if len(toLXe)>0:
        a=(-np.sqrt(d[:, toLXe])-np.sum(us[:, toLXe]*vs[:, toLXe], axis=0)) # N len array
        vs[:, toLXe]=vs[:, toLXe]+us[:, toLXe]*a
        rot=np.cross(us[:, toLXe], smr(-vs[:, toLXe], sigma), axis=0)
        rot=rot/np.sqrt(np.sum(rot**2, axis=0))
        inn=np.pi-np.arccos(np.sum(vs[:, toLXe]*us[:, toLXe], axis=0)/np.sqrt(np.sum(vs[:, toLXe]**2, axis=0))) # N len array
        TIR=np.nonzero(np.sin(inn)*nHPFS/nLXe>1)[0]
        Rif=np.nonzero(np.sin(inn)*nHPFS/nLXe<=1)[0]

        if len(Rif)>0:
            out=np.arcsin(np.sin(inn[Rif])*nHPFS/nLXe) # N len array
            theta=inn[Rif]-out
            us[:,toLXe[Rif]]=rot[:,Rif]*np.sum(rot[:,Rif]*us[:,toLXe[Rif]], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,Rif], us[:,toLXe[Rif]], axis=0),
                rot[:,Rif], axis=0)+np.sin(theta)*np.cross(rot[:,Rif], us[:,toLXe[Rif]], axis=0)

        if len(TIR)>0:
            theta=-(np.pi-2*inn[TIR])
            us[:,ToLXe[TIR]]=rot[:, TIR]*np.sum(rot[:,TIR]*us[:,ToLXe[TIR]], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,TIR], us[:,TIR], axis=0),
                rot[:,TIR], axis=0)+np.sin(theta)*np.cross(rot[:,TIR], us[:,ToLXe[TIR]], axis=0)
    if len(toHPFS)>0:
        vs[:,toHPFS], us[:,toHPFS]=tracetoVac(vs[:,toHPFS], us[:,toHPFS])
    us=us/np.sqrt(np.sum(us*us, axis=0))
    return vs+1e-6*us, us


def tracetoVac(vs, us, nLXe, sigma):
    nHPFS=1.6
    # us and vs is an (3,N) array
    a=(np.sqrt(np.sum(vs*us, axis=0)**2+(0.75**2-np.sum(vs**2, axis=0)))-np.sum(us*vs, axis=0)) # N len array
    vs=vs+us*a
    rot=np.cross(us, smr(vs, sigma), axis=0)
    rot=rot/np.sqrt(np.sum(rot**2, axis=0))
    inn=np.arccos(np.sum(vs*us, axis=0)/np.sqrt(np.sum(vs**2, axis=0))) # N len array
    TIR=np.nonzero(np.sin(inn)*nHPFS>1)[0]
    Rif=np.nonzero(np.sin(inn)*nHPFS<=1)[0]

    if len(Rif)>0:
        out=np.arcsin(np.sin(inn[Rif])*nHPFS) # N len array
        theta=inn[Rif]-out
        us[:,Rif]=rot[:,Rif]*np.sum(rot[:,Rif]*us[:,Rif], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,Rif], us[:,Rif], axis=0), rot[:,Rif], axis=0)+np.sin(theta)*np.cross(rot[:,Rif], us[:,Rif], axis=0)

    if len(TIR)>0:
        theta=-(np.pi-2*inn[TIR])
        us[:,TIR]=rot[:,TIR]*np.sum(rot[:,TIR]*us[:,TIR], axis=0)+np.cos(theta)*np.cross(np.cross(rot[:,TIR], us[:,TIR], axis=0), rot[:,TIR], axis=0)+np.sin(theta)*np.cross(rot[:,TIR], us[:,TIR], axis=0)
    us=us/np.sqrt(np.sum(us*us, axis=0))

    return vs+1e-6*us, us
