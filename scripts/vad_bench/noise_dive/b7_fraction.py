"""Mechanism: share of the normalisation window that is noise. 120 s clip = (120-L) s of a real talk + L s of noise in the middle."""
import numpy as np,soundfile as sf,json
from vadlib import *; from sig import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
x=sf.read('talk/JaneMcGonigal-merged.wav',dtype='float32')[0][60*SR:300*SR]
rdb=20*np.log10(rms(x)); r=np.random.default_rng(2)
Ls=[5,10,20,30,45,60,90]; rels=[-35,-25,-15,-5]; R={}
fig,axs=plt.subplots(2,2,figsize=(11,7),sharey=True)
for mi,m in enumerate(['redux','ultra']):
    for ki,kind in enumerate(['white','pink']):
        ax=axs[mi][ki]
        for rel in rels:
            ys=[]
            for L in Ls:
                fp=[]
                for rep in range(3):
                    s0=int(r.integers(0,100))*SR; sp=120-L; a=sp//2
                    pre=x[s0:s0+a*SR]; post=x[s0+a*SR:s0+sp*SR]
                    n=at_db(white(L*SR,r) if kind=='white' else colored(L*SR,r,1),rdb+rel)
                    y=np.concatenate([pre,n,post]).astype(np.float32)
                    z=ref(m).logits(y); t=(np.arange(len(z))+.5)*.08; st=(t>=a+1)&(t<a+L-1); fp.append((z[st]>=0).mean())
                ys.append(100*np.mean(fp)); R[f'{m}|{kind}|{rel}|{L}']=ys[-1]
            ax.plot([100*L/120 for L in Ls],ys,marker='o',label=f'noise {rel} dB re talk RMS')
        ax.set_title(f'{m}, {kind} noise stretch'); ax.set_xlabel('noise share of the 120 s normalisation window (%)'); ax.set_ylabel('% stretch frames called speech'); ax.grid(alpha=.3)
axs[0][0].legend(fontsize=8); plt.tight_layout(); plt.savefig('fig_noise_fraction.png',dpi=110)
print('talk rms dBFS %.1f'%rdb)
for m in ['redux','ultra']:
    for kind in ['white','pink']:
        print(f'\n{m} {kind}: stretch FA % ; rows noise level re talk RMS, cols noise share of window (L s of 120)'); print('| level | '+' | '.join(f'{L} s ({100*L//120}%)' for L in Ls)+' |'); print('|---|'+'---|'*len(Ls))
        for rel in rels: print(f'| {rel} dB | '+' | '.join(f'{R[f"{m}|{kind}|{rel}|{L}"]:.0f}' for L in Ls)+' |')
json.dump(R,open('res_B7.json','w'))
