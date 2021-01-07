from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.stats import poisson, binom
from scipy.special import factorial, comb, erf, expi
from scipy.special import comb


path='../../cluster/arrays/'
path=''
data=np.load('q.npz')
ps=data['ps']
qs=data['qs']
ls=data['ls']
# step=data['step']
ps=ps[:len(ls)]
qs=qs[:len(ls)]

print(len(ls))
id=np.nonzero(~np.isnan(ls))[0]
ps=ps[id]
qs=qs[id]
ls=ls[id]
# step=step[id]

# step=step[ls<1e6]
ps=ps[ls<1e7]
qs=qs[ls<1e7]
ls=ls[ls<1e7]
# step=step[ls>0]
# ps=ps[ls>0]
# ls=ls[ls>0]
# div=data['div']
print(np.mean(ls[-50:]), np.std(ls[-50:]))
print(list(np.round(np.mean(ps[-50:], axis=0), decimals=5)))
for i in range(5):
    print(np.mean(qs[-50:, i,0]), np.std(qs[-50:, i,0]))
# for i in range(5):
#     print(np.mean(qs[-50:, i,1]), np.std(qs[-50:, i,1]))

names=['Q0', 'Q1', 'Q2', 'Q3','Q4','Q5', 'Q6', 'Q7', 'Q8', 'Q9','Q10','Q11', 'Q12', 'Q13', 'Q14',
        'St0', 'St1', 'St2', 'St3','St4','St5', 'St6', 'St7', 'St8', 'St9','St10','St11', 'St12', 'St13', 'St14',
        'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9', 'fano', 'nLXe', 'sigma_smr', 'mu', 'R1','R2','R3','R4','R5','R6','R7','R8','R9',
         'a', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'Tf', 'Ts']

print(len(ps[0]))
for i in range(len(ps[0])):
    # print(i, names[i])
    plt.figure()
    plt.title(names[i]+'({})'.format(i))
    plt.plot(ps[:,i], 'ko')
    plt.plot(np.argmin(ls), ps[np.argmin(ls),i], 'ro')


print(np.argmin(ls), np.amin(ls))
fig, ax=plt.subplots(1)
ax.plot(np.arange(len(ls)), ls, 'ko', label='{}'.format(np.mean(ls[-50:]))+r'$\pm$'+'{}'.format(np.std(ls[-50:])))
ax.plot(np.argmin(ls), ls[np.argmin(ls)], 'ro')
# ax.bar(np.arange(len(ls))[step=='r'], ls[step=='r'], width=1, color='r')
# ax.bar(np.arange(len(ls))[step=='e'], ls[step=='e'], width=1, color='g')
# ax.bar(np.arange(len(ls))[step=='c'], ls[step=='c'], width=1, color='c')
# ax.bar(np.arange(len(ls))[step=='q'], ls[step=='q'], width=1, color='y')
# ax.bar(np.arange(len(ls))[step=='i'], ls[step=='i'], width=1, color='b')


#
# for i in range(5):
#     ax.plot(qs[:,i,0]/div[i], 'o', label='{}'.format(i))
    # ax.plot(qs[:,i,1]/div[i+5], 'o', label='{}'.format(i))

ax.legend()
ax.set_yscale('log')

plt.show()
