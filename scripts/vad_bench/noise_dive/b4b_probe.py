import os
FUSION=os.environ.get('FUSION_DIR','../fusion')  # scripts/vad_bench/fusion with data/ built by prep_libri.py
"""Feature-space probe: feed the head (via the real subsampler) normalised-mel inputs that we construct directly."""
import numpy as np,json
from vadlib import *; from sig import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng=np.random.default_rng(8); N=30*SR; T=3000
lib=list(np.load(FUSION+'/data/libri.npy',allow_pickle=True))
sp=np.concatenate(lib[10:16]).astype(np.float32)[:N]
res={}
mel_sp=norm_mel(raw_logmel(sp))                      # speech as the net sees it
mel_wn=norm_mel(raw_logmel(at_db(white(N,rng),-25)))
iid=torch.tensor(rng.standard_normal((T,128)).astype(np.float32))   # iid N(0,1) features, no structure at all
print('A) iid N(0,1) normalised features (no structure) -> speech frames %')
for m in ['redux','ultra']:
    r=ref(m); print(' ',m,100*(r.head_logits(r.sub_out(iid))>=0).float().mean().item(),'% ; mean logit',r.head_logits(r.sub_out(iid)).mean().item(), '; all-zero features logit',r.head_logits(r.sub_out(torch.zeros(T,128))).mean().item())
print('B) offset d (in per-file std units) added to every bin and frame; scale s on the fluctuation (white noise normalised mel)')
ds=np.arange(-3,3.01,0.5); ss=[0.25,0.5,1.0,1.5,2.0]
fig,ax=plt.subplots(1,2,figsize=(11,4))
for mi,m in enumerate(['redux','ultra']):
    r=ref(m); Mi=np.zeros((len(ss),len(ds)))
    for i,s in enumerate(ss):
        for j,d in enumerate(ds):
            x=mel_wn*s+d; Mi[i,j]=(r.head_logits(r.sub_out(x))>=0).float().mean().item()
    res[f'offset_scale|{m}']=Mi.tolist()
    im=ax[mi].imshow(100*Mi,origin='lower',aspect='auto',extent=[ds[0]-.25,ds[-1]+.25,-0.5,len(ss)-.5],vmin=0,vmax=100,cmap='viridis')
    ax[mi].set_yticks(range(len(ss))); ax[mi].set_yticklabels(ss); ax[mi].set_xlabel('offset d (std units of the file)'); ax[mi].set_ylabel('fluctuation scale s'); ax[mi].set_title(f'{m}: % frames speech, white-noise features*s+d')
    plt.colorbar(im,ax=ax[mi])
    print(m); print('   s\\d '+' '.join(f'{d:5.1f}' for d in ds))
    for i,s in enumerate(ss): print(f'  {s:4.2f} '+' '.join(f'{100*v:5.0f}' for v in Mi[i]))
plt.tight_layout(); plt.savefig('fig_offset_scale.png',dpi=110); plt.close()
# C) speech file: shift the whole normalised mel by d
print('C) real speech normalised mel shifted by d: speech frames % (of frames)')
for m in ['redux','ultra']:
    r=ref(m); print(' ',m,[ (d,round(100*(r.head_logits(r.sub_out(mel_sp+d))>=0).float().mean().item())) for d in [-2,-1,-0.5,0,0.5,1,2]])
json.dump(res,open('res_B4b.json','w'))
