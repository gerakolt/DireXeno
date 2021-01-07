import multiprocessing
import numpy as np
import sys
import time

def make_glob_array(p, n, m):
    Q=p[:n]
    St=p[n:2*n]
    W=p[2*n:2*n+m]
    fano=p[2*n+m]
    nLXe=p[2*n+m+1]*0+1.62
    sigma_smr=p[2*n+m+2]*0+2
    mu=p[2*n+m+3]*0+100
    R=p[2*n+m+4:2*n+2*m+4]
    a=p[2*n+2*m+4]
    F=p[2*n+2*m+5:2*n+3*m+5]
    Tf=p[2*n+3*m+5]
    Ts=p[2*n+3*m+6]
    return Q, St, W, fano, nLXe, sigma_smr, mu, R, a, F, Tf, Ts


def make_iter(N, Q, St, Sa, nLXe, sigma_smr, R, a, F, Tf, Ts, v):
    for i in range(len(N)):
        yield [N[i], Q, St, Sa, nLXe, sigma_smr, R, a, F, Tf, Ts, v[:,i], i]


def make_ps():
    n=2*15+3*9+7
    ps=np.zeros((n+1, n))
    ps[:,:15]=np.random.uniform(0.15, 0.25, size=len(np.ravel(ps[:,:15]))).reshape(np.shape(ps[:,:15]))
    ps[:,15:30]=np.random.uniform(0.1, 1, size=len(np.ravel(ps[:,:15]))).reshape(np.shape(ps[:,:15]))
    ps[:,30:39]=np.random.uniform(1, 25, size=len(np.ravel(ps[:,30:35]))).reshape(np.shape(ps[:,30:35]))
    ps[:,39]=np.random.uniform(0.01, 0.5, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    ps[:,40]=np.random.uniform(1.55, 1.72, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))*0+1.62
    ps[:,41]=np.random.uniform(0.01, 1, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))*0+2
    ps[:,42]=np.random.uniform(0.01, 2, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))*0+100
    ps[:,43:25]=np.random.uniform(0.1, 0.9, size=len(np.ravel(ps[:,39:44]))).reshape(np.shape(ps[:,39:44]))
    ps[:,52]=np.random.uniform(1e-3, 9e-2, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    ps[:,53:62]=np.random.uniform(0.5, 1, size=len(np.ravel(ps[:,45:50]))).reshape(np.shape(ps[:,45:50]))
    ps[:,62]=np.random.uniform(0.8, 2.5, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    ps[:,63]=np.random.uniform(15, 50, size=len(np.ravel(ps[:,6]))).reshape(np.shape(ps[:,6]))
    return ps

# def make_ps2(p0, a):
#     n=2*15+3*9+7
#     ps=np.zeros((n+25, n))
#     ps[0]=np.array(p0)
#     for i, p in enumerate(p0):
#         if i in [45,46,47,48,49, 39,40,41,42,43]:
#             if p*(1+a)>=1:
#                 ps[1:,i]=np.random.uniform(p-a*p,1, size=np.shape(ps)[0]-1)
#             else:
#                 ps[1:,i]=np.random.uniform(p-a*p,p+a*p, size=np.shape(ps)[0]-1)
#         else:
#             ps[1:,i]=np.random.uniform(p-a*p,p+a*p, size=np.shape(ps)[0]-1)
#     return ps

def make_ps2(p0, a):
    n=2*15+3*9+7
    ps=np.zeros((n+25, n))
    ps[0]=np.array(p0)
    for i, p in enumerate(p0):
        if i<39 and i>=30:
            ps[1:,i]=np.random.uniform(1,15, size=np.shape(ps)[0]-1)
        else:
            ps[1:,i]=np.random.uniform(p,p, size=np.shape(ps)[0]-1)
    return ps

def make_ps3(p0, a):
    n=2*15+3*6+7
    ps=np.zeros((n+25, n))
    ps[0]=np.array(p0)
    for i, p in enumerate(p0):
        ps[1:,i]=np.random.uniform(p,p, size=np.shape(ps)[0]-1)
    return ps
