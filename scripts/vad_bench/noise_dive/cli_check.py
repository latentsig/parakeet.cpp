"""Cross-check with the shipped parakeet-cli on a few rebuilt clips: gap clips (embedded stretch) and amplitude."""
import numpy as np,soundfile as sf,subprocess,json,os
from vadlib import *; from sig import *; import corpus
os.makedirs('wav',exist_ok=True); gaps=corpus.gap_clips(); res=[]
for g in gaps:
    if g['k'] not in (0,1) or g['snr'] not in (20,5,0) or g['kind']!='white': continue
    wr(f"wav/{g['name']}.wav",g['y']); n=g['name']
    for m,gg in [('redux','redux'),('ultra','ultra')]:
        c=cli_probs(gg,f'wav/{n}.wav'); z=ref(m).logits(sf.read(f'wav/{n}.wav',dtype='float32')[0]); p=sig(z)
        t=(np.arange(len(c))+.5)*.08; st=(t>=g['stretch'][0])&(t<g['stretch'][1])
        print(n,m,'stretch FP pct CLI %.1f ref %.1f | max|dp| %.4f'%(100*(c[st]>=.5).mean(),100*(p[:len(c)][st]>=.5).mean(),np.abs(c-p[:len(c)]).max()),flush=True)
r=np.random.default_rng(1); b=white(30*SR,r)
for lv in [-20,-50,-60,-70,-80]:
    wr(f'wav/amp_{lv}.wav',at_db(b,lv))
    print('amp',lv,{m:round(100*(cli_probs(m,f'wav/amp_{lv}.wav')>=.5).mean(),1) for m in ['redux','ultra']},flush=True)
