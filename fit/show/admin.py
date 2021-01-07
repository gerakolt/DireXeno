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


def make_ps(p0):
    ps=np.zeros((10, len(p0)))
    for i, p in enumerate(p0):
        ps[:,i]=p
    return ps
