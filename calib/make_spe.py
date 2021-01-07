import numpy as np
import time
import matplotlib.pyplot as plt


pmts=np.array([0,1,2,4,6,7,3,10,13,15,17,18,5,11,12,14])
chns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
PMT_num=17
time_samples=1024
path='/home/gerak/Desktop/DireXeno/301120B/PulserA/'

pmt=18
data=np.load(path+'PMT{}/cuts.npz'.format(pmt))
blw_cut=data['blw_cut']
height_cut=data['height_cut']
left=data['left']
right=data['right']
BL=np.load(path+'PMT{}/BL.npz'.format(pmt))['BL']
# BL=np.zeros(1000)

file=open(path+'out.DXD', 'rb')

id=0
j=0
start_time = time.time()
rec=np.recarray(100000, dtype=[
    ('height', 'i8'),
    ('blw', 'f8'),
    ('area', 'f8'),
    ('init10', 'i8'),
    ('id', 'i8'),
    ('rise_time', 'i8'),
    ('dh3', 'f8'),
    ('maxi', 'i8'),
    ('spe', 'f8', 1000)
    ])
while id<1e5:
    if id%100==0:
        print('Event number {} ({} files per sec). {} file were saved.'.format(id, 100/(time.time()-start_time), j))
        start_time = time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    # trig=find_trig(Data[2:1002,PMT_num+2])
    trig=np.argmin(Data[2:1002, 0])
    wf=Data[2:1002, chns[np.nonzero(pmts==pmt)[0][0]]]
    wf=np.roll(wf, 100-trig)
    bl=np.median(wf[:100])
    wf-=bl+BL
    blw=np.sqrt(np.mean((wf[:100])**2))
    rec[j]['blw']=blw
    h=-np.amin(wf[left:right])
    rec[j]['height']=h
    maxi=np.argmin(wf)
    if maxi<100 or maxi>799:
        rec[j]['area']=-1e4
    else:
        rec[j]['area']=-(np.sum(wf[maxi-100:maxi+200])+np.sum(wf[maxi-50:maxi+150])+np.sum(wf[maxi-100:maxi+150])+np.sum(wf[maxi-50:maxi+200]))/4
    rec[j]['maxi']=maxi
    if len(np.nonzero(wf[:maxi]>-0.1*h)[0])>0:
        init10=np.amax(np.nonzero(wf[:maxi]>-0.1*h)[0])
    else:
        init10=0
    rec[j]['rise_time']=maxi-init10
    rec[j]['dh3']=(h+wf[maxi-3])/h
    rec[j]['init10']=init10
    rec[j]['id']=id

    if h<height_cut or maxi<left or maxi>right or init10==0:
        rec[j]['spe']=np.zeros(1000)
    else:
        rec[j]['spe']=np.roll(wf, 200-init10)

    j+=1
    id+=1
np.savez(path+'PMT{}/spe'.format(pmt), rec=rec[:j-1])
