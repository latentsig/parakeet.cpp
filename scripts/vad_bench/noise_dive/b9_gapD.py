import numpy as np,json
from vadlib import *; import corpus
rows=[]
for g in corpus.gap_clips():
    y=g['y']; mn=norm_mel(raw_logmel(y)); a,b=g['stretch']; D=float(mn[int(a*100):int(b*100)].mean())
    for m in ['redux','ultra']:
        z=logits_blocks(m,y); t=(np.arange(len(z))+.5)*.08; st=(t>=a)&(t<b)
        rows.append(dict(m=m,kind=g['kind'],snr=g['snr'],D=D,fa=float((z[st]>=0).mean()),med=float(np.median(z[st]))))
json.dump(rows,open('res_B9.json','w'))
for m in ['redux','ultra']:
    print(m)
    for snr in [20,10,5,0]:
        P=[r for r in rows if r['m']==m and r['snr']==snr]; print(f'  SNR {snr:2d}: mean D {np.mean([r["D"] for r in P]):+.2f}  FA {100*np.mean([r["fa"] for r in P]):.0f}%  median logit {np.mean([r["med"] for r in P]):+.2f}')
