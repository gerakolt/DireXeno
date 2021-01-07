import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib.colors as mcolors

path='/home/gerak/Desktop/DireXeno/221120/'

Data = np.genfromtxt(path+'scan.out', delimiter=',')
th=Data[:,0]
R=Data[:,4:20]


pmts=[0,1,2,4,5,6,7,8,10,11,13,15,17,18,12,14]
fig, ax=plt.subplots(2,2)
amp=[4]
for i in range(16):
    np.ravel(ax)[i//4].axvline(50, 0 ,1, color='k')
    if pmts[i] in amp:
        marker='+'
    else:
        marker='o'
    np.ravel(ax)[i//4].plot(th, R[:,i], marker, label='PMT{}'.format(pmts[i]))
    np.ravel(ax)[i//4].set_yscale('log')
    np.ravel(ax)[i//4].legend()
plt.show()
