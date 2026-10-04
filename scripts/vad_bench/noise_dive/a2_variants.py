"""Does an alternative head wiring (the documented debug variants) change the noise behaviour? (rules out 'wrong activation/residual' as the cause)"""
import numpy as np,torch,json
from vadlib import *; from sig import *; import corpus
F=torch.nn.functional
def head(r,s,act_proj,res,act_ctx):
    a=lambda k,x: F.silu(x) if k=='silu' else F.relu(x)
    h=a(act_proj,s@r.pw.T+r.pb); c=F.conv1d(h.T[None],r.cw,r.cb,padding=2)[0].T
    if res: c=c+h
    return (a(act_ctx,c)@r.ow+r.ob).numpy()
V=[('silu','silu',False),('relu','relu',False),('silu','silu',True),('relu','relu',True)]
cl=corpus.clean_clips()[:20]; gaps=[g for g in corpus.gap_clips() if g['snr'] in(5,0)][:12:3]+[g for g in corpus.gap_clips() if g['snr'] in(5,0)][48:60:3]
rng=np.random.default_rng(5); N=30*SR
nz=[at_db(white(N,rng),-23),at_db(colored(N,rng,1),-23),at_db(white(N,rng),-40),at_db(colored(N,rng,1),-40)]
for m in ['redux','ultra']:
    r=ref(m); print(f'\n{m}: variant (proj act, ctx act, ctx residual) | clean speech F1 % | speech frames in noise-only -23 dBFS % | -40 dBFS % | embedded 5/0 dB stretch FP %')
    sub_cl=[(r.sub_out(norm_mel(raw_logmel(y))),sp) for y,sp in cl]
    sub_nz=[r.sub_out(norm_mel(raw_logmel(x))) for x in nz]
    sub_g=[(r.sub_out(norm_mel(raw_logmel(g['y']))),g) for g in gaps]
    for ap,ac,res in V:
        c=np.zeros(3)
        for s,sp in sub_cl:
            z=head(r,s,ap,res,ac); p=z>=0; ref_m=frame_mask(sp,len(p)); c+=[(p&ref_m).sum(),(p&~ref_m).sum(),(~p&ref_m).sum()]
        P=c[0]/(c[0]+c[1]); R=c[0]/(c[0]+c[2])
        n23=np.mean([(head(r,s,ap,res,ac)>=0).mean() for s in sub_nz[:2]]); n40=np.mean([(head(r,s,ap,res,ac)>=0).mean() for s in sub_nz[2:]])
        fps=[]
        for s,g in sub_g:
            z=head(r,s,ap,res,ac); t=(np.arange(len(z))+.5)*.08; fps.append((z[(t>=g['stretch'][0])&(t<g['stretch'][1])]>=0).mean())
        print(f'  ({ap},{ac},res={res}) | F1 {100*2*P*R/(P+R):.1f} | {100*n23:.0f} | {100*n40:.0f} | {100*np.mean(fps):.0f}',flush=True)
