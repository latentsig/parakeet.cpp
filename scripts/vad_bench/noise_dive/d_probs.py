import numpy as np,json,torch,soundfile as sf
from vadlib import *
S=np.load('fixed_stats.npz'); fa=(torch.tensor(S['aM']),torch.tensor(S['aS']))
meta=json.load(open('talk/d/meta.json')); a,b=meta['ins']; res={}
print('| file | model | insert frames called speech %: block-120 s norm (shipped) | whole-file norm | fixed norm A | median logit (shipped) |\n|---|---|---|---|---|---|')
for f,tn,rel in meta['files']:
    x=sf.read(f'talk/d/{f}.wav',dtype='float32')[0]
    for m in ['redux','ultra']:
        zb=logits_blocks(m,x); zf=ref(m).logits(x); zx=ref(m).logits(x,*fa)
        t=(np.arange(len(zb))+.5)*.08; st=(t>=a+1)&(t<b-1)
        if f=='base': st=(t>=a-30)&(t<a+30)   # base: report a 60 s window at the same place (the original talk audio) as control
        res[f'{f}|{m}']=dict(block=float((zb[st]>=0).mean()),whole=float((zf[:len(zb)][st]>=0).mean()),fixed=float((zx[:len(zb)][st]>=0).mean()),med=float(np.median(zb[st])))
        r=res[f'{f}|{m}']; print(f'| {f} | {m} | {100*r["block"]:.0f} | {100*r["whole"]:.0f} | {100*r["fixed"]:.0f} | {r["med"]:+.1f} |',flush=True)
json.dump(res,open('res_D_probs.json','w'))
