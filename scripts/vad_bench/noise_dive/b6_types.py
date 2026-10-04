import numpy as np,json
from vadlib import *; from sig import *; import corpus
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
models=['redux','ultra']; fold_of=corpus.folds(); libri=corpus.libri; spk=corpus.spk
spec=ltass(libri[:60])
def mk(k, noise_fn, snr, whole):
    f=k%4; idx=[i for i in range(len(libri)) if fold_of[spk[i]]==f]; r=np.random.default_rng(900+k); idx=list(r.permutation(idx))[:5]
    us=[libri[i] for i in idx]
    y1,s1=corpus.FL.build(us[:2],np.random.default_rng(400+k)); y2,s2=corpus.FL.build(us[2:],np.random.default_rng(500+k))
    off=len(y1)/16000+30.0
    y=np.concatenate([y1,np.random.default_rng(600+k).standard_normal(30*16000)*1e-3,y2]).astype(np.float32)
    spans=s1+[(a+off,b+off) for a,b in s2]
    P=np.mean(np.concatenate([y[int(a*SR):int(b*SR)] for a,b in spans])**2)
    st=(s1[-1][1]+0.5,s2[0][0]+off-0.5); a,b=int(st[0]*SR),int(st[1]*SR)
    nr=np.random.default_rng(77+k)
    if whole:
        n=noise_fn(len(y),nr); n=n/ (rms(n)+1e-12)*np.sqrt(P/10**(snr/10)); y=y+n.astype(np.float32)
    else:
        n=noise_fn(b-a,nr); n=n/(rms(n)+1e-12)*np.sqrt(P/10**(snr/10)); y=y.copy(); y[a:b]+=n.astype(np.float32)
    return np.clip(y,-1,1).astype(np.float32),st
types={'white':(lambda n,r:white(n,r),True),'pink':(lambda n,r:colored(n,r,1),True),'brown':(lambda n,r:colored(n,r,2),True),'speech-shaped':(lambda n,r:shaped(n,r,spec),True),
 'hum 50 Hz':(lambda n,r:hum(n,50,-10),False),'sine 1 kHz':(lambda n,r:sine(n,1000,-10),False),'clicks 3/s':(lambda n,r:clicks(n,r,3,-10),False),'clicks 10/s':(lambda n,r:clicks(n,r,10,-10),False),'music-like':(lambda n,r:music(n,r,-10),False)}
snrs=[-10,0,10,20,30,40]
R={}
print('Embedded-stretch false-alarm frame % at threshold 0.5 ; rows=type, cols=SNR re speech (dB) ; 6 clips per cell')
for m in models:
    print(f'\n{m}:'); print('| type | '+' | '.join(f'{s} dB' for s in snrs)+' |'); print('|---|'+'---|'*len(snrs))
    for tn,(fn,whole) in types.items():
        row=[]
        for s in snrs:
            fps=[];mz=[]
            for k in range(6):
                y,st=mk(k,fn,s,whole); z=ref(m).logits(y); t=(np.arange(len(z))+.5)*.08; msk=(t>=st[0])&(t<st[1]); fps.append((z[msk]>=0).mean()); mz.append(np.median(z[msk]))
            R[f'{m}|{tn}|{s}']=(float(np.mean(fps)),float(np.mean(mz))); row.append(f'{100*np.mean(fps):.0f}')
        print(f'| {tn} | '+' | '.join(row)+' |',flush=True)
json.dump(R,open('res_B6.json','w'))
fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
for ax,m in zip(axs,models):
    for tn in types: ax.plot(snrs,[100*R[f'{m}|{tn}|{s}'][0] for s in snrs],marker='o',label=tn,ls='-' if types[tn][1] else '--')
    ax.set_title(f'{m}: stretch false alarms vs level of the stretch below speech'); ax.set_xlabel('stretch level re speech power (dB below speech)'); ax.set_ylabel('% stretch frames called speech'); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig('fig_types_vs_level.png',dpi=110)
