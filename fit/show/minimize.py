import numpy as np
import multiprocessing
from L import L, make_Ravel
from admin import make_ps




p0=[0.19383, 0.33869, 0.15957, 0.29701, 0.19992, 0.26311, 0.29221, 0.30597, 0.363, 0.24516, 0.18231, 0.2643, 0.29809, 0.40271, 0.23644,
 0.49821, 2.32831, 0.39712, 0.44029, 0.78527, 1.20807, 1.39166, 1.02096, 1.61056, 1.11271, 0.86397, 1.65534, 0.298, 0.41841, 0.63702, 3.67509, 4.66585,
  4.62691, 5.45184, 4.01384, 3.8029, 6.12232, 5.38578, 5.53983, 0.15047, 0.79543, 0.90712, 9.81462,
  0.24036, 0.24232, 0.25, 0.26719, 0.4574, 0.54, 0.6, 0.73311, 0.74406,
    11.43112,
    0.93013, 0.9, 0.81669, 0.67115, 0.6, 0.52581, 0.40951, 0.30378, 0.31024,
    0.44933, 34.00794]




def minimize(pmt, q, ID):
    stop=0
    step=[]
    ls=[]
    Qs=np.zeros((1000, 5, 1))
    Ravel_Spectrum, Ravel_Spectra, Ravel_G, Ravel_H, Ravel_Angs, Ravel_Angs10, Ravel_Fullspectrum, Ravel_W=make_Ravel()
    ps=make_ps(p0)
    PS=np.zeros((1000, np.shape(ps)[1]))
    l=np.zeros(np.shape(ps)[0])
    for i in range(len(l)):
        step.append('i')
        l[i]=L(ps[i], pmt, q, PS, ls, Qs, ID, step, Ravel_Spectrum, Ravel_Spectra, Ravel_G, Ravel_H, Ravel_Angs, Ravel_Angs10, Ravel_Fullspectrum, Ravel_W)
