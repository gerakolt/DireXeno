import numpy as np
import time
import matplotlib.pyplot as plt



PMT_num=17
time_samples=1024
path='/home/gerak/Desktop/DireXeno/301120B/PulserA/'
file=open(path+'out.DXD', 'rb')

pmts=[0,1,2,4,6,7,3,10,13,15,17,18,5,11,12,14]
chns=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

rec=np.recarray(100000, dtype=[
    ('blw', 'f8', len(pmts)),
    ('trig', 'f8'),
    ('height', 'i8', len(pmts)),
    ('maxi', 'i8', len(pmts)),
    ('bl', 'i8', len(pmts)),
    ])

id=0
j=0
start_time = time.time()
while id<1e5:
    if id%100==0:
        print('Event number {} ({} files per sec). {} file were saved.'.format(id, 100/(time.time()-start_time), j))
        start_time = time.time()
    Data=np.fromfile(file, np.float32, (PMT_num+4)*(time_samples+2))
    if len(Data)<(PMT_num+4)*(time_samples+2):
        break
    Data=np.reshape(Data, (PMT_num+4, time_samples+2)).T
    #trig=find_trig(Data[2:1002,0])
    trig=np.argmin(Data[2:1002, 0])
    Trig=Data[2:1002, 0]
    Trig-=np.median(Trig[:100])
    rec['trig'][j]=trig
    for i, pmt in enumerate(pmts):
        wf_temp=Data[2:1002, chns[i]]
        wf=np.roll(wf_temp, 100-trig)
        bl=np.median(wf[:100])
        wf=wf-bl
        blw=np.sqrt(np.mean(wf[:100]**2))
        # if np.amin(wf)<-30:
        #     plt.figure()
        #     plt.plot(wf, 'r--')
        #     plt.plot(wf_temp-bl, 'k+')
        #     plt.show()
        rec[j]['blw'][i]=blw
        rec[j]['height'][i]=-np.amin(wf)
        rec[j]['maxi'][i]=np.argmin(wf)
        rec[j]['bl'][i]=bl
    j+=1
    id+=1

fig, ax=plt.subplots(4,4)
for i in range(16):
    np.ravel(ax)[i].hist(rec['bl'][:j,i])

np.savez(path+'raw_wf'.format(pmt), rec=rec[:j])
plt.hist(rec['trig'][:j], bins=100)
plt.show()
