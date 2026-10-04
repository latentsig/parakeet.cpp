"""Two-variable predictor of the head's response on a noise stretch: D (mean normalised level of the stretch) and S (temporal std of the
normalised log-mel inside the stretch, mean over bins = fluctuation scale s of the feature-space probe). Pool talk+inserted-noise clips (b8) and the fusion gap clips."""
import numpy as np,soundfile as sf,json
from vadlib import *; from sig import *; import corpus
from sklearn.linear_model import LinearRegression
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pts=[]
def add(y,a,b,src,kind):
    mn=norm_mel(raw_logmel(y)).numpy(); seg=mn[int((a+1)*100):int((b-1)*100)]; D=float(seg.mean()); S=float(seg.std(0).mean())
    for m in ['redux','ultra']:
        z=ref(m).logits(y); t=(np.arange(len(z))+.5)*.08; st=(t>=a+1)&(t<b-1)
        pts.append(dict(m=m,src=src,kind=kind,D=D,S=S,fa=float((z[st]>=0).mean()),med=float(np.median(z[st]))))
for g in corpus.gap_clips():
    if g['k']<6: add(g['y'],g['stretch'][0],g['stretch'][1],'gap',g['kind'])
x=sf.read('talk/JaneMcGonigal-merged.wav',dtype='float32')[0][60*SR:300*SR]; rdb=20*np.log10(rms(x)); r=np.random.default_rng(12)
for kind in ['white','pink']:
    for rel in [-35,-25,-15,-5]:
        for L in [10,30,60]:
            s0=int(r.integers(0,100))*SR; sp=120-L; a=sp//2
            n=at_db(white(L*SR,r) if kind=='white' else colored(L*SR,r,1),rdb+rel)
            y=np.concatenate([x[s0:s0+a*SR],n,x[s0+a*SR:s0+sp*SR]]).astype(np.float32); add(y,a,a+L,'talk+insert',kind)
json.dump(pts,open('res_B10.json','w'))
for m in ['redux','ultra']:
    P=[p for p in pts if p['m']==m]; X=np.array([[p['D'],p['S']] for p in P]); M=np.clip(np.array([p['med'] for p in P]),-10,10)
    for nm,cols in [('D only',[0]),('S only',[1]),('D and S',[0,1])]:
        lr=LinearRegression().fit(X[:,cols],M); print(m,nm,'R2 of median logit: %.2f'%lr.score(X[:,cols],M),'coef',lr.coef_.round(2))
    for src in ['gap','talk+insert']:
        Q=[p for p in P if p['src']==src]; print(f'   {src:12s} n={len(Q)} mean D {np.mean([p["D"] for p in Q]):+.2f} mean S {np.mean([p["S"] for p in Q]):.2f} FA {100*np.mean([p["fa"] for p in Q]):.0f}% median logit {np.mean([p["med"] for p in Q]):+.2f}')
fig,axs=plt.subplots(1,2,figsize=(12,4.2),sharey=True)
for ax,m in zip(axs,['redux','ultra']):
    for src,c in [('gap','C3'),('talk+insert','C0')]:
        Q=[p for p in pts if p['m']==m and p['src']==src]; sc=ax.scatter([p['D'] for p in Q],[p['S'] for p in Q],c=[p['med'] for p in Q],cmap='coolwarm',vmin=-6,vmax=3,marker='o' if src=='gap' else 's',s=30,edgecolors='k',linewidths=.4,label=src)
    ax.set_xlabel('D: mean normalised level of the stretch'); ax.set_title(f'{m}: colour = median logit in the stretch'); ax.legend(fontsize=8)
axs[0].set_ylabel('S: temporal std of normalised features in the stretch'); plt.colorbar(sc,ax=axs.ravel().tolist()); plt.savefig('fig_D_S.png',dpi=110,bbox_inches='tight')
