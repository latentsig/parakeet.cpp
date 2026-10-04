"""Unifying predictor: normalised level of the stretch, D = mean over stretch frames and mel bins of the per-file-normalised log-mel
(0 = the file's average level, -1 = one per-bin std below it). Relate D to the head's decision on the stretch."""
import numpy as np,soundfile as sf,json
from vadlib import *; from sig import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
x=sf.read('talk/JaneMcGonigal-merged.wav',dtype='float32')[0][60*SR:300*SR]; rdb=20*np.log10(rms(x)); r=np.random.default_rng(12)
pts=[]
for kind in ['white','pink','speech-shaped','music']:
    for rel in [-35,-30,-25,-20,-15,-10,-5]:
        for L in [10,30,60]:
            for rep in range(2):
                s0=int(r.integers(0,100))*SR; sp=120-L; a=sp//2
                n=at_db(white(L*SR,r) if kind=='white' else colored(L*SR,r,1) if kind=='pink' else shaped(L*SR,r,ltass([x[:SR*60]])) if kind=='speech-shaped' else music(L*SR,r,0),rdb+rel)
                y=np.concatenate([x[s0:s0+a*SR],n,x[s0+a*SR:s0+sp*SR]]).astype(np.float32)
                mn=norm_mel(raw_logmel(y)); fr0=int((a+1)*100); fr1=int((a+L-1)*100); D=float(mn[fr0:fr1].mean())
                for m in ['redux','ultra']:
                    z=ref(m).logits(y); t=(np.arange(len(z))+.5)*.08; st=(t>=a+1)&(t<a+L-1)
                    pts.append(dict(m=m,kind=kind,rel=rel,L=L,D=D,fa=float((z[st]>=0).mean()),med=float(np.median(z[st]))))
json.dump(pts,open('res_B8.json','w'))
fig,axs=plt.subplots(1,2,figsize=(12,4.2),sharey=True)
for ax,m in zip(axs,['redux','ultra']):
    for kind,c in zip(['white','pink','speech-shaped','music'],['C0','C1','C3','C2']):
        P=[p for p in pts if p['m']==m and p['kind']==kind]; ax.scatter([p['D'] for p in P],[p['med'] for p in P],s=18,c=c,label=kind,alpha=.7)
    ax.axhline(0,color='k',ls=':'); ax.set_xlabel('normalised level of the stretch D (std units; 0 = file average)'); ax.set_title(f'{m}: median logit in a noise stretch vs D'); ax.legend(fontsize=8); ax.grid(alpha=.3)
axs[0].set_ylabel('median logit in stretch'); plt.tight_layout(); plt.savefig('fig_logit_vs_delta.png',dpi=110)
for m in ['redux','ultra']:
    P=[p for p in pts if p['m']==m]; D=np.array([p['D'] for p in P]); M=np.array([p['med'] for p in P]); F=np.array([p['fa'] for p in P])
    print(m,'corr(D, median logit) %.2f ; corr(D, FA) %.2f'%(np.corrcoef(D,M)[0,1],np.corrcoef(D,F)[0,1]))
    for lo,hi in [(-9,-2),(-2,-1.5),(-1.5,-1),(-1,-.5),(-.5,0),(0,.5),(.5,9)]:
        s=(D>=lo)&(D<hi)
        if s.sum(): print(f'   D in [{lo},{hi}): n={s.sum():3d} FA mean {100*F[s].mean():5.1f}%  median logit {np.median(M[s]):+.2f}')
