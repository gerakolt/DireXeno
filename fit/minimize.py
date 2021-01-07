import numpy as np
import multiprocessing
from L import L, make_Ravel
from admin import make_ps, make_ps2, make_ps3




p0=[0.19383, 0.33869, 0.15957, 0.29701, 0.19992, 0.26311, 0.29221, 0.30597, 0.363, 0.24516, 0.18231, 0.2643, 0.29809, 0.40271, 0.23644,
 0.7607137 ,  0.70664638,  0.70108794,  0.7081122,   0.71983791,  0.74337253,  0.6421211,
  0.74077786 , 0.74203365,  0.75295714 , 0.69932017,  0.71032887,  0.82576841,  0.78990368 , 0.68267526,
  4.35517, 5, 6.06718, 5,5, 5.05112,
   0.15047, 0.79543, 0.90712, 9.81462,
    0.24036, 0.24232, 0.25, 0.26719, 0.4574, 0.54, 11.43112,
     0.93013, 0.9, 0.81669, 0.67115, 0.6, 0.52581, 0.44933, 34.00794]


def minimize(pmt, q, ID):
    stop=0
    step=[]
    ls=[]
    Qs=np.zeros((1000, 5, 1))
    Ravel_Spectrum, Ravel_Spectra, Ravel_G, Ravel_H, Ravel_Angs, Ravel_Angs10, Ravel_Fullspectrum, Ravel_W, Ravel_KeVPE=make_Ravel()

    # ps, l=import_ps('q.npz')
    ps=make_ps3(p0, 0.23)
    PS=np.zeros((1000, np.shape(ps)[1]))
    l=np.zeros(np.shape(ps)[0])
    for i in range(len(l)):
        step.append('i')
        l[i]=L(ps[i], pmt, q, PS, ls, Qs, ID, step, Ravel_KeVPE, Ravel_Spectrum, Ravel_Spectra, Ravel_G, Ravel_H, Ravel_Angs, Ravel_Angs10,
                        Ravel_Fullspectrum, Ravel_W)

    while not stop:
        stop=1
        count=0
        if pmt>=0:
            ps[:,pmt]=q
        while len(ls)<1550:
            count+=1
            h=0.5
            a=1
            g=1
            s=0.5
            ind=np.argsort(l)
            m=np.mean(ps[ind[:-1]], axis=0)
            r=m+a*(m-ps[ind[-1]])
            step.append('r')
            lr=L(r, pmt, q, PS, ls, Qs, ID, step, Ravel_KeVPE, Ravel_Spectrum, Ravel_Spectra, Ravel_G, Ravel_H, Ravel_Angs, Ravel_Angs10, Ravel_Fullspectrum, Ravel_W)
            if l[ind[0]]<lr and lr<l[ind[-2]]:
                ps[ind[-1]]=r
                l[ind[-1]]=lr
            elif lr<l[ind[0]]:
                e=m+g*(r-m)
                step.append('e')
                le=L(e, pmt, q, PS, ls, Qs, ID, step, Ravel_KeVPE, Ravel_Spectrum, Ravel_Spectra, Ravel_G, Ravel_H, Ravel_Angs, Ravel_Angs10, Ravel_Fullspectrum, Ravel_W)
                if le<lr:
                    ps[ind[-1]]=e
                    l[ind[-1]]=le
                else:
                    ps[ind[-1]]=r
                    l[ind[-1]]=lr
            else:
                c=m+h*(ps[ind[-1]]-m)
                step.append('c')
                lc=L(c, pmt, q, PS, ls, Qs, ID, step, Ravel_KeVPE, Ravel_Spectrum, Ravel_Spectra, Ravel_G, Ravel_H, Ravel_Angs, Ravel_Angs10, Ravel_Fullspectrum, Ravel_W)
                if lc<l[ind[-1]]:
                    ps[ind[-1]]=c
                    l[ind[-1]]=lc
                else:
                    for i in ind[1:]:
                        ps[i]=ps[ind[0]]+s*(ps[i]-ps[ind[0]])
                        step.append('q')
                        l[i]=L(ps[i], pmt, q, PS, ls, Qs, ID, step, Ravel_KeVPE, Ravel_Spectrum, Ravel_Spectra, Ravel_G, Ravel_H, Ravel_Angs, Ravel_Angs10, Ravel_Fullspectrum, Ravel_W)


def import_ps(name):
    data=np.load(name)
    ps=data['ps']
    ls=data['ls']
    # p=ps[np.argsort(ls)[:100]]
    return ps[np.argsort(ls)], np.sort(ls)
