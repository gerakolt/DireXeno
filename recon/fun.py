import numpy as np
from classes import Hit, Group
from scipy.stats import poisson, binom
from scipy.optimize import minimize
import sys
try:
    import matplotlib.pyplot as plt
except:
    pass
from scipy import fft, ifft



def get_spes(pmts):
    spes=[]
    height_cuts=[]
    dh3_cut=[]
    BL=[]
    for pmt in pmts:
        try:
            path='/home/gerak/Desktop/DireXeno/Pulser/PMT{}/'.format(pmt)
            data=np.load(path+'cuts.npz')
        except:
            path='/storage/xenon/gerak/301120B/Pulser/PMT{}/'.format(pmt)
            data=np.load(path+'cuts.npz')
        dh3_cut.append(data['dh3_cut'])
        height_cuts.append(data['height_cut'])
        data=np.load(path+'SPE.npz')
        spe=data['spe']
        spe[np.argmax(spe)+180:]=0
        spes.append(spe)
        # BL.append(np.load(path+'BL.npz')['BL'])
        BL.append(np.zeros(1000))
    return spes, height_cuts, dh3_cut, BL



def Recon_wf(WF, wf_origin, height_cut, dh3_cut, SPE, Init):
    area_spe=np.sum(SPE)
    t=[]
    recon_wf=np.zeros(1000)
    wf=np.array(wf_origin)
    dif=(wf-np.roll(wf,1))
    dif[0]=dif[1]
    dif_bl=np.median(dif[:Init])
    dif_blw=np.sqrt(np.mean((dif[:Init]-dif_bl)**2))
    SPE_dif=(SPE-np.roll(SPE,1))
    SPE_dif[0]=SPE_dif[1]
    maxis=Init+np.nonzero(np.logical_and(wf[Init:990]<-height_cut,
                    np.logical_and(np.logical_and(wf[Init:990]<=np.roll(wf[Init:990], -1), wf[Init:990]<np.roll(wf[Init:990], -2)),
                    np.logical_and(wf[Init:990]<=np.roll(wf[Init:990], 1), wf[Init:990]<np.roll(wf[Init:990], 2)))))[0]
    maxis=maxis[np.logical_and((wf[maxis]-wf[maxis-3])/wf[maxis]<dh3_cut, (wf[maxis]-wf[maxis+3])/wf[maxis]<dh3_cut)]
    counter=0
    while len(maxis)>0 and counter<1000:
        counter+=1
        maxi=maxis[0]
        if len(np.nonzero(np.logical_and(wf[:maxi]>-WF.blw, dif[:maxi]>dif_bl-dif_blw))[0])>0:
            left=np.amax(np.nonzero(np.logical_and(wf[:maxi]>-WF.blw, dif[:maxi]>dif_bl-dif_blw))[0])
        else:
            left=0
        if len(np.nonzero(np.logical_and(wf[maxi:]>-WF.blw, dif[maxi:]>dif_bl-dif_blw))[0])>0:
            right=maxi+np.amin(np.nonzero(np.logical_and(wf[maxi:]>-WF.blw, dif[maxi:]>dif_bl-dif_blw))[0])
        else:
            right=len(wf)-1

        area_frac=np.sum(wf[left:right])/area_spe
        a=np.exp(-0.5*(area_frac-1)**2/(0.5)**2)
        b=np.exp(-0.5*(area_frac-2)**2/(2*(0.5)**2))
        if np.amin(SPE)<np.amin(wf[left:right]):
            I=left+np.argmin(wf[left:right])
            if I>50 and I<999-75:
                spe=np.roll(np.amin(wf[left:right])*SPE/(np.amin(SPE)), I-np.argmin(SPE))
                ffty=np.abs(fft(wf[I-50:I+75]))
                fftSPE=np.abs(fft(spe[I-50:I+75]))
                ffty=ffty/ffty[0]
                fftSPE=fftSPE/fftSPE[0]
                p=np.sqrt(np.sum((ffty[:20]-fftSPE[:20])**2))
                if p<0.8:
                    t.append(I)
                else:
                    spe=np.zeros(1000)
                    spe[left:right]+=wf[left:right]*(height_cut/np.amin(wf[left:right])+1)
            else:
                spe=np.zeros(1000)
                spe[left:right]+=wf[left:right]*(height_cut/np.amin(wf[left:right])+1)

        elif area_frac<2 and np.random.choice([0,1], p=[a/(a+b), b/(a+b)])==0:
            I=left+np.argmin(wf[left:right])
            if I==left:
                if WF.blw>height_cut:
                    spe=np.zeros(1000)
                    spe[left:right]+=wf[left:right]*(height_cut/np.amin(wf[left:right])+1)
                else:
                    print('Dont know')
                    break
            else:
                spe=np.roll(SPE, I-np.argmin(SPE))
                spe=np.amin(wf[left:np.argmin(spe)])*spe/(np.amin(spe))
                t.append(I)

        else:
            Chi2=1e9
            I=np.argmin(SPE)
            for i in range(left+10, maxi+10):
                spe=np.roll(SPE, i-np.argmin(SPE))
                spe_dif=np.roll(SPE_dif, i-np.argmin(SPE_dif))
                chi2=np.sum((spe[left:i-5]-wf[left:i-5])**2)+np.sum((spe_dif[left:i-5]-dif[left:i-5])**2)
                if chi2<Chi2:
                    Chi2=chi2
                    I=i
            spe=np.roll(SPE, I-np.argmin(SPE))
            t.append(I)
        #
        # J=np.argmin(spe)
        # ffty=np.abs(fft(wf[J-50:J+75]))
        # fftSPE=np.abs(fft(spe[J-50:J+75]))
        # ffty=ffty/ffty[0]
        # fftSPE=fftSPE/fftSPE[0]
        # p=np.sqrt(np.sum((ffty[:20]-fftSPE[:20])**2))


        spe_dif=np.roll(SPE_dif, I-np.argmin(SPE_dif))
        wf[left:]-=spe[left:]
        wf[wf>0]=0
        recon_wf[left:]+=spe[left:]
        dif=(wf-np.roll(wf,1))
        dif[0]=dif[1]

        # temp=np.zeros(1000)
        # temp[left:]=wf[left:]+spe[left:]
        # x=np.arange(1000)/5
        # plt.figure()
        # plt.title(counter)
        # plt.plot(x, spe, 'b-',label='SPE Template', linewidth=3)
        # plt.plot(x, temp, 'g--',label='Residual Signal', linewidth=3)
        # plt.plot(x, recon_wf, 'r--', label='Reconstructed Signal', linewidth=3)
        # plt.plot(x, wf, 'y--', label='NEw', linewidth=3)
        # plt.axhline(-height_cut, 0, 1, color='c', linewidth=3)
        # plt.fill_between(x[left:right], wf_origin[left:right], 0, color='k', alpha=0.3, label='Original Signal')
        # plt.fill_between(x[J-50:J+75], wf_origin[J-50:J+75], 0, color='k', alpha=0.3)
        # plt.plot(x, wf_origin, 'k--', linewidth=3, label=np.round(area_frac, decimals=3))
        # plt.xlabel('Time [ns]', fontsize=25)
        # plt.ylabel('Digitizer Counts', fontsize=25)
        # plt.legend(fontsize=35)
        # plt.tick_params(axis='both', which='major', labelsize=20)
        #
        # plt.figure()
        # plt.plot(np.abs(ffty), '.-')
        # plt.plot(np.abs(fftSPE), 'r--', label=np.round(p, decimals=3))
        # plt.legend()
        # plt.show()


        maxis=Init+np.nonzero(np.logical_and(wf[Init:990]<-height_cut, np.logical_and(wf[Init:990]<np.roll(wf[Init:990], -1),
         wf[Init:990]<np.roll(wf[Init:990], 1))))[0]
        maxis=maxis[np.logical_and((wf[maxis]-wf[maxis-3])/wf[maxis]<dh3_cut, (wf[maxis]-wf[maxis+3])/wf[maxis]<dh3_cut)]
    return np.histogram(t, bins=np.arange(1001))[0], recon_wf


def find_hits(self, wf_origin, Init, height_cut, rise_time_cut):
    wf=np.array(wf_origin)
    dif=(wf-np.roll(wf,1))
    dif[0]=dif[1]
    dif_bl=np.median(dif[:Init])
    dif_blw=np.sqrt(np.mean((dif[:Init]-dif_bl)**2))

    while np.amin(wf)<-height_cut:
        maxi=np.argmin(wf)
        if len(np.nonzero(np.logical_and(wf[:maxi]>-self.blw, dif[:maxi]>dif_bl-dif_blw))[0])>0:
            left=np.amax(np.nonzero(np.logical_and(wf[:maxi]>-self.blw, dif[:maxi]>dif_bl-dif_blw))[0])
        else:
            left=0
        if len(np.nonzero(np.logical_and(wf[maxi:]>-self.blw, dif[maxi:]>dif_bl-dif_blw))[0])>0:
            right=maxi+np.amin(np.nonzero(np.logical_and(wf[maxi:]>-self.blw, dif[maxi:]>dif_bl-dif_blw))[0])
        else:
            right=len(wf)
        if maxi-left>rise_time_cut:
            hit=Hit(left, right)
            hit.area=-np.sum(wf[left:right])
            self.hits.append(hit)
        wf[left:right]=0
