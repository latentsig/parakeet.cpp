# Build the fusion corpus: 4 speaker folds x 10 clips x 9 conditions (+ noise-only clips). Speakers never cross folds.
import json,os,numpy as np,soundfile as sf
from lib import *
libri=list(np.load("data/libri.npy",allow_pickle=True)); spk=np.load("data/spk.npy")
rng=np.random.default_rng(11)
sp_ids=rng.permutation(sorted(set(spk))); fold_of={s:i%4 for i,s in enumerate(sp_ids)}
sets=[]  # (fold, utts)
for f in range(4):
    idx=[i for i in range(len(libri)) if fold_of[spk[i]]==f]; idx=list(rng.permutation(idx))
    for k in range(min(10,len(idx)//5)): sets.append((f,[libri[i] for i in idx[k*5:(k+1)*5]]))
conds=[("clean",None)]+[(k,s) for k in("white","pink") for s in(20,10,5,0)]
os.makedirs("clips",exist_ok=True); meta=[]; Ps=[]
for ci,(kind,snr) in enumerate(conds):
    for k,(f,us) in enumerate(sets):
        y,spans=build(us,np.random.default_rng(1000+k))
        if ci==0: Ps.append(float(np.mean(np.concatenate([y[int(s*16000):int(e*16000)] for s,e in spans])**2)))
        y=add_noise(y,spans,kind,snr,np.random.default_rng(5000+ci*100+k))
        name=f"{kind}{'' if snr is None else snr}_{k:02d}"
        sf.write(f"clips/{name}.wav",y,16000,subtype="PCM_16")
        meta.append(dict(name=name,cond=name.split("_")[0],fold=f,ref=spans,dur=len(y)/16000))
# noise-only: 12 x 30 s per noise condition, level from a speech set's power (same SNR definition)
for ci,(kind,snr) in enumerate(conds[1:]):
    for k in range(12):
        r=np.random.default_rng(9000+ci*100+k); P=Ps[(k*3)%len(Ps)]; n=30*16000
        z=r.standard_normal(n) if kind=="white" else pink(n,r)
        z=np.clip(z*np.sqrt(P/10**(snr/10)),-1,1).astype(np.float32)
        name=f"nz-{kind}{snr}_{k:02d}"; sf.write(f"clips/{name}.wav",z,16000,subtype="PCM_16")
        meta.append(dict(name=name,cond=f"nz-{kind}{snr}",fold=-1,ref=[],dur=30.0))
json.dump(meta,open("clips.json","w")); print(len(meta),sum(m["dur"] for m in meta))
