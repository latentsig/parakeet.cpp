import os
FUSION=os.environ.get('FUSION_DIR','../fusion')  # scripts/vad_bench/fusion with data/ built by prep_libri.py
import numpy as np,json,soundfile as sf,sys
sys.path.insert(0,FUSION); import fl
from vadlib import *
meta=json.load(open('talk/d/meta.json')); a,b=meta['ins']
def runconf(z,q):
    keep=np.zeros(len(z),bool)
    for i,j in zip(*fl.runs(z>=0)):
        if np.median(z[i:j])>=q: keep[i:j]=True
    return keep
print('insert (60 s) frames called speech %, by mitigation (block-normalised logits, shipped segmenter smoothing not applied)\n| file | model | thr 0.5 | thr 0.7 | thr 0.9 | runconf 2.5 | talk speech frames kept (outside insert, vs thr 0.5) runconf 2.5 |\n|---|---|---|---|---|---|---|')
for f,tn,rel in meta['files']:
    if f=='base': continue
    x=sf.read(f'talk/d/{f}.wav',dtype='float32')[0]
    for m in ['redux','ultra']:
        z=logits_blocks(m,x); t=(np.arange(len(z))+.5)*.08; st=(t>=a+1)&(t<b-1); out=(t<a-1)|(t>b+1)
        rc=runconf(z,2.5); base=z>=0
        print(f'| {f} | {m} | {100*base[st].mean():.0f} | {100*(z[st]>=np.log(.7/.3)).mean():.0f} | {100*(z[st]>=np.log(.9/.1)).mean():.0f} | {100*rc[st].mean():.0f} | {100*rc[out].sum()/base[out].sum():.1f}% |',flush=True)
