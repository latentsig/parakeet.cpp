#!/usr/bin/env python3
"""Synthetic benchmark: Parakeet VAD head (Ultra Q8_0, Redux packed) vs Silero.

usage: PARAKEET_LIB=build/libparakeet.so compare_synthetic.py DATA_DIR GGUF_DIR OUT.json [CLIPS_PER_COND]
GGUF_DIR holds ultra-q8_0.gguf and redux-keep.gguf. Needs: numpy soundfile torch silero-vad onnxruntime.
Writes the raw spans of every system; analyze_synthetic.py turns them into the tables.
"""
import json,sys,numpy as np,soundfile as sf
from heads_lib import *
SYS={}
def make_systems():
    S={"silero_matched":lambda y:silero(y,True),"silero_default":lambda y:silero(y,False)}
    return S
DATA,G,OUT=sys.argv[1],sys.argv[2],sys.argv[3]   # data dir, gguf dir, output json
N_CLIPS=int(sys.argv[4]) if len(sys.argv)>4 else 24   # clips per condition
libri=list(np.load(DATA+"/libri.npy",allow_pickle=True))
rng=np.random.default_rng(7)
order=rng.permutation(len(libri))
sets=[[libri[i] for i in order[k*5:(k+1)*5]] for k in range(len(libri)//5)][:N_CLIPS]
conds=[("clean",None)]+[(k,s) for k in("white","pink") for s in(20,10,5,0)]
# build corpus once
corpus=[]
for ci,(kind,snr) in enumerate(conds):
    for k,us in enumerate(sets):
        r=np.random.default_rng(1000+k)            # same speech layout per clip across conditions
        y,spans=build(us,r)
        y=add_noise(y,spans,kind,snr,np.random.default_rng(5000+ci*100+k))
        corpus.append((f"{kind}{'' if snr is None else snr}",y,spans))
print("corpus",len(corpus),"wavs",sum(len(c[1]) for c in corpus)/16000,"s",flush=True)
models={"ultra_q8":"ultra-q8_0.gguf","redux_packed":"redux-keep.gguf"}
preds={}   # system -> list of spans per corpus item
for name,fn in make_systems().items():
    preds[name]=[fn(c[1]) for c in corpus]
print("silero done",flush=True)
for mname,f in models.items():
    pk=PK(G+"/"+f)
    preds[mname]=[parakeet(pk,c[1]) for c in corpus]
    # threshold sweep on a subset of conditions
    for thr in (0.3,0.7):
        preds[f"{mname}@{thr}"]=[parakeet(pk,c[1],thr) if c[0] in("clean","white10","white0","pink10") else None for c in corpus]
    print(mname,"done",flush=True)
# silero thresholds
for thr in (0.3,0.7):
    preds[f"silero_matched@{thr}"]=[None]*len(corpus)
    for i,c in enumerate(corpus):
        if c[0] in("clean","white10","white0","pink10"):
            ts=get_speech_timestamps(torch.from_numpy(c[1]),SIL,sampling_rate=16000,return_seconds=True,threshold=thr,min_speech_duration_ms=100,min_silence_duration_ms=200,speech_pad_ms=0)
            preds[f"silero_matched@{thr}"][i]=[(t["start"],t["end"]) for t in ts]
json.dump({"conds":[c[0] for c in corpus],"refs":[c[2] for c in corpus],"preds":preds},open(OUT,"w"))
print("saved",flush=True)
