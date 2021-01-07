import numpy as np
import sys

pmts=[0,1,4,6,7,3,10,13,15,17,18,5,11,12,14]

def make_pmts(PMTs):
    r=21/40
    pmt_mid=np.array([[1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), 1/np.sqrt(2)],
                    [-1/np.sqrt(2), 0, 1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), 1/np.sqrt(2)],
                    [0,1,0],
                    [-1/np.sqrt(2), 1/np.sqrt(2), 0],
                    [-1,0,0],
                    [-1/np.sqrt(2), -1/np.sqrt(2),0],
                    [1/np.sqrt(2)*np.cos(120*np.pi/180), 1/np.sqrt(2)*np.sin(120*np.pi/180), -1/np.sqrt(2)],
                    [-1/np.sqrt(2), 0, -1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(240*np.pi/180), 1/np.sqrt(2)*np.sin(240*np.pi/180), -1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), 1/np.sqrt(2)],
                    [1/np.sqrt(2), 0,1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), 1/np.sqrt(2)],
                    [0, -1, 0],
                    [1/np.sqrt(2), -1/np.sqrt(2), 0],
                    [1, 0, 0],
                    [1/np.sqrt(2), 1/np.sqrt(2), 0],
                    [1/np.sqrt(2)*np.cos(300*np.pi/180), 1/np.sqrt(2)*np.sin(300*np.pi/180), -1/np.sqrt(2)],
                    [1/np.sqrt(2), 0, -1/np.sqrt(2)],
                    [1/np.sqrt(2)*np.cos(60*np.pi/180), 1/np.sqrt(2)*np.sin(60*np.pi/180), -1/np.sqrt(2)],
                    ])


    pmt_r=np.zeros((len(PMTs),3))
    pmt_up=np.zeros((len(PMTs),3))

    for i, pmt in enumerate(PMTs):
        a=pmt_mid[pmt][0]
        b=pmt_mid[pmt][1]
        pmt_r[i]=[-0.5*r*b/np.sqrt(a**2+b**2), 0.5*r*a/np.sqrt(a**2+b**2), 0]
        pmt_up[i]=np.cross(pmt_mid[pmt], pmt_r[i])
    return pmt_mid[PMTs], pmt_r, pmt_up

def make_mash(pmts):
    X=[0]
    Y=[0]
    Z=[0]

    r=0.75
    for i in range(8):
        X.append(r*np.cos(i*np.pi/4))
        Y.append(r*np.sin(i*np.pi/4))
        Z.append(0)
    for i in range(6):
        X.append(r*np.cos(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
        Y.append(r*np.sin(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
        Z.append(r*np.cos(np.pi/4))
    for i in range(6):
        X.append(r*np.cos(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
        Y.append(r*np.sin(i*np.pi/3+np.pi/6)*np.sin(np.pi/4))
        Z.append(-r*np.cos(np.pi/4))
    X.append(0)
    Y.append(0)
    Z.append(-r)
    X.append(0)
    Y.append(0)
    Z.append(r)
    r=np.vstack((X, np.vstack((Y,Z)))).T/4
    return r


mid, rt, up=make_pmts(pmts)
r_mash=make_mash(pmts)
def make_dS(d):
    r=np.sqrt(np.sum(rt[0]**2))
    dS=np.zeros((len(r_mash), len(pmts)))
    a=np.linspace(-1,1,100, endpoint=True)
    I=np.arange(len(pmts)*len(a)**2*3)
    V=np.zeros(len(r_mash))
    for i in range(len(d)):
        print('in make dS', i, 'out of', len(d))
        V[np.argmin(np.sum((r_mash-d[i])**2, axis=-1))]+=1
        u=(mid[I//(3*len(a)**2), I%3]+a[(I//3)%len(a)]*rt[I//(3*len(a)**2), I%3]+a[(I//(3*len(a)))%len(a)]*up[I//(3*len(a)**2), I%3]-d[i,I%3]).reshape(len(pmts),len(a)**2, 3)
        u3=(np.sqrt(np.sum(u**2, axis=-1)))**3
        sum_u3=((a[1]-a[0])*r)**2*np.sum(1/u3, axis=-1)
        md=np.sum(mid*d[i], axis=-1)
        dS[np.argmin(np.sum((r_mash-d[i])**2, axis=-1))]+=(1-md)*sum_u3
    dS=(dS.T/V).T
    return V/len(d), dS/(4*np.pi)


def make_K(pmts, Angs):
    ang=[]
    K=np.zeros((len(pmts), len(pmts))).astype(int)
    for i in range(len(pmts)):
        for j in range(len(pmts)):
            ang.append(np.arccos(np.sum(mid[i]*mid[j]))/np.pi*180)
            K[i,j]=np.argmin(np.abs(np.arccos(np.sum(mid[i]*mid[j]))/np.pi*180-Angs))
    h, bins=np.histogram(ang, bins=np.arange(182)-0.5)
    U=h[h>0]/np.sum(h[h>0])
    return K, U



def whichPMT(vs, us):
    # v is 3,N
    hits=np.zeros(len(us[0]))-1
    for i in range(len(mid)):
        a=(1-np.sum(vs.T*mid[i], axis=1))/np.sum(us.T*mid[i], axis=1) # N length
        r=(vs+a*us).T-mid[i] # (N,3)
        hits[np.nonzero(np.logical_and(a>0, np.logical_and(np.abs(np.sum(r*rt[i], axis=1))<np.sum(rt[i]**2),
         np.abs(np.sum(r*up[i], axis=1))<np.sum(up[i]**2))))[0]]=i
    return hits
