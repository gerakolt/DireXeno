import numpy as np
import matplotlib.pyplot as plt
import time
from classes import WaveForm
from fun import find_hits, Recon_wf, get_spes
import sys

PMT_num=17
time_samples=1024
id=34050
pmts=[0,1,4,6,7,3,10,13,15,17,18, 5, 11, 12, 14]
chns=[1,2,4,5,6,7,8,9,10,11,12, 13,14,15,16]
Init=1
spes, height_cuts, rise_time_cuts, BL=get_spes(pmts)

#313
#27817 28710 33746 34050 34125 34751

delays=[ 0.   ,      -0.02383257 , 0.30002359,  1.39715157, -0.34396877 , 0.76709905,
 -0.92513445, -0.44346783 , 0.74364364,  0.83153115 , 0.72306022, -0.1138911,
 -0.08735925, -2.32931312 , 0.38945054 ,-1.10135374]

WFs=np.zeros((len(pmts), 1000))
recon_WFs=np.zeros((len(pmts), 1000))

init_event=1000
path='/home/gerak/Desktop/DireXeno/011220/'
file=open(path+'out.DXD', 'rb')
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
trig=np.median(Data[2:1002,0])
H=np.zeros(1000)
fig, ax=plt.subplots(4,4)
fig, ax1=plt.subplots(4,4)


Area=0
for i, pmt in enumerate(pmts):
    # if not pmt==6:
    #     continue
    wf=Data[2:1002, chns[i]]
    wf=wf-np.median(wf[:100])
    blw=np.sqrt(np.mean(wf[:100]**2))
    init_wf=1001
    area=-1
    for k in range(np.argmin(wf)):
        if np.all(wf[k:k+20]<-blw):
            init_wf=k
            area=np.sum(wf[k:])/np.sum(spes[i][np.argmin(spes[i])-100:np.argmin(spes[i])+200])
            break
    Area+=area
    wf=np.roll(wf, -int(np.round(delays[i]*5)))
    waveform=WaveForm(blw)
    h, recon_wf=Recon_wf(waveform, wf, height_cuts[i], rise_time_cuts[i], spes[i], Init)
    chi2=np.sqrt(np.sum((wf[Init:]-recon_wf[Init:])**2))
    if np.sum(h)>0:
        chi2pe=chi2/np.sum(h)
    else:
        chi2pe=0
    np.ravel(ax1)[i].step(np.arange(1000)/5, h, where='mid')
    H+=h
    np.ravel(ax)[i].plot(wf, 'k-+', label='PMT: {}'.format(pmts[i]))
    np.ravel(ax)[i].plot(recon_wf, 'r-.', label='Resolved PEs: {}'.format(np.sum(h)))
    np.ravel(ax)[i].plot(spes[i], 'y--')
    np.ravel(ax)[i].legend()
plt.show()
init=np.amin(np.nonzero(H>0)[0])
print(np.sum(H))
for i, pmt in enumerate(pmts):
    wf=Data[2:1002, chns[i]]
    wf=wf-np.median(wf[:100])
    blw=np.sqrt(np.mean(wf[:100]**2))
    wf=np.roll(wf, -int(np.round(delays[i]*5)))
    np.ravel(ax)[i].fill_between(np.arange(1000)[init:init+8*5], wf[init:init+8*5], color='y', alpha=0.5)
    np.ravel(ax)[i].fill_between(np.arange(1000)[init+8*5:init+16*5], wf[init+8*5:init+16*5], color='g', alpha=0.5)
plt.figure()
plt.step(np.arange(1000), H, where='mid')
plt.show()
