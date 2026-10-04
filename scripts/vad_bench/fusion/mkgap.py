# "Embedded noise stretch" clips: 2 utterances, 30 s of noise-only, 3 utterances; noise over the whole clip at the given SNR.
# Tests false alarms where the file also contains speech (the head normalises features per file).
import json,numpy as np,soundfile as sf
from lib import *
libri=list(np.load("data/libri.npy",allow_pickle=True)); spk=np.load("data/spk.npy")
rng=np.random.default_rng(11)
sp_ids=rng.permutation(sorted(set(spk))); fold_of={s:i%4 for i,s in enumerate(sp_ids)}
meta=[]
conds=[(k,s) for k in("white","pink") for s in(20,10,5,0)]
for ci,(kind,snr) in enumerate(conds):
    for k in range(12):
        f=k%4; idx=[i for i in range(len(libri)) if fold_of[spk[i]]==f]; r=np.random.default_rng(300+ci*50+k); idx=list(r.permutation(idx))[:5]
        us=[libri[i] for i in idx]
        y1,s1=build(us[:2],np.random.default_rng(400+k)); y2,s2=build(us[2:],np.random.default_rng(500+k))
        off=len(y1)/16000+30.0
        y=np.concatenate([y1,np.random.default_rng(600+k).standard_normal(30*16000)*1e-3,y2]).astype(np.float32)
        spans=s1+[(a+off,b+off) for a,b in s2]
        y=add_noise(y,spans,kind,snr,np.random.default_rng(7000+ci*100+k))
        name=f"gap-{kind}{snr}_{k:02d}"; sf.write(f"clips/{name}.wav",y,16000,subtype="PCM_16")
        meta.append(dict(name=name,cond=f"{kind}{snr}",fold=f,ref=spans,dur=len(y)/16000,stretch=[s1[-1][1]+0.5,s2[0][0]+off-0.5]))
json.dump(meta,open("gap.json","w")); print(len(meta),sum(m["dur"] for m in meta))
