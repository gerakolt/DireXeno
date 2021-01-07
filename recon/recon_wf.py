import numpy as np
import time
from classes import WaveForm
# import matplotlib.pyplot as plt

from fun import find_hits, Recon_wf, get_spes
import sys

PMT_num=17
time_samples=1024
id=0
pmts=[0,1,2,4,6,7,3,10,13,15,17,18,5,11,12,14]
chns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
Init=20
spes, height_cuts, rise_time_cuts, BL=get_spes(pmts)

delays=[ 0.     ,    -0.06196179, 0.3 ,  1.39596335, -0.23545325,  0.7899076,
 -0.86156509, -0.49753459,  0.47695548,  0.77041075,  0.52034361, -0.10062916,
 -0.08752133, -2.31063218 , 0.37934098, -1.0219721 ]

for n in range(4,5):
    WFs=np.zeros((len(pmts), 1000))
    recon_WFs=np.zeros((len(pmts), 1000))

    start_time = time.time()
    rec=np.recarray(1000, dtype=[
        ('area', 'f8', len(pmts)),
        ('height', 'f8', len(pmts)),
        ('blw', 'f8', len(pmts)),
        ('id', 'i8'),
        ('time', 'i8'),
        ('chi2', 'f8', len(pmts)),
        ('h', 'i8', (1000, len(pmts))),
        ('init_event', 'i8'),
        ('NGtrig', 'i8'),
        ('init_wf', 'i8', len(pmts))
        ])

    try:
        path='/home/gerak/Desktop/DireXeno/011220/'
        file=open(path+'out2.DXD', 'rb')
    except:
        path='/storage/xenon/gerak/011220/NG{}/'.format(n)
        file=open(path+'out.DXD', 'rb')

    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2)*id)
    j=0
    NGcount=0
    while True:
        if id%100==0:
            print(id, (time.time()-start_time)/100, 'sec per events')
            print(path, pmts)
            start_time=time.time()
        Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
        if len(Data)<(PMT_num+4)*(time_samples+2):
            break
        Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
        trig=np.median(Data[2:1002,0])
        if trig<500 or NGcount>1:
            rec[j]['id']=id
            rec[j]['time']=Data[0,0]
            if trig<500:
                NGcount+=1
                rec[j]['NGtrig']=0
            else:
                NGcount-=2
                rec[j]['NGtrig']=1
            H=np.zeros(1000)

            for i, pmt in enumerate(pmts):
                wf=Data[2:1002, chns[i]]
                wf=wf-np.median(wf[:100])
                blw=np.sqrt(np.mean(wf[:100]**2))
                init_wf=1001
                area=-1
                for k in range(np.argmin(wf)):
                    if np.all(wf[k:k+20]<-blw):
                        init_wf=k
                        area=np.sum(wf[k:])/np.sum(spes[i])
                        break
                rec[j]['init_wf'][i]=init_wf
                rec[j]['area'][i]=area
                rec[j]['height'][i]=-np.amin(wf)
                wf=np.roll(wf, -int(np.round(delays[i]*5)))
                waveform=WaveForm(blw)
                h, recon_wf=Recon_wf(waveform, wf, height_cuts[i], rise_time_cuts[i], spes[i], Init)
                H+=h
                chi2=np.sqrt(np.sum((wf[Init:]-recon_wf[Init:])**2))
                if blw<30 and rec[j]['init_wf'][i]>100 and chi2<5000:
                    WFs[i]+=wf
                    recon_WFs[i]+=recon_wf

                rec[j]['blw'][i]=blw
                rec[j]['chi2'][i]=chi2
                rec[j]['h'][:,i]=h


            if len(np.nonzero(H>0)[0])==0:
                init=-1
            else:
                init=np.amin(np.nonzero(H>0)[0])
            rec[j]['init_event']=init
            for i, pmt in enumerate(pmts):
                rec[j]['h'][:,i]=np.roll(rec[j]['h'][:,i], -init)
            j+=1

        id+=1
        if j==len(rec):
            np.savez(path+'EventRecon/recon1ns{}'.format(id-1), rec=rec, WFs=WFs, recon_WFs=recon_WFs, pmts=pmts)
            WFs=np.zeros((len(pmts), 1000))
            recon_WFs=np.zeros((len(pmts), 1000))
            j=0
    np.savez(path+'EventRecon/recon1ns{}'.format(id-1), rec=rec[:j-1], WFs=WFs, recon_WFs=recon_WFs, pmts=pmts)
