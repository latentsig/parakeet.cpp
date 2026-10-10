#!/usr/bin/env python3
"""Non-speech audio: MUSAN music (every k-th file), MUSAN noise, ESC-50 (every 4th clip) -> 16k mono wav files (noise/esc concatenated into ~10 min files)."""
import io, json, os, sys
import numpy as np, soundfile as sf, librosa
ROOT=os.environ.get("VAD_REAL_ROOT") or os.getcwd()
os.environ.setdefault("HF_HOME",f"{ROOT}/hfhome")
from datasets import Audio, load_dataset
which=sys.argv[1]; outd=sys.argv[2]
def dec(ex):
    y,sr=sf.read(io.BytesIO(ex["audio"]["bytes"]),dtype="float32")
    if y.ndim>1: y=y.mean(axis=1)
    if sr!=16000: y=librosa.resample(y,orig_sr=sr,target_sr=16000)
    return y
if which=="music":
    k=int(sys.argv[3]); cap=float(sys.argv[4]); tot=0; start=int(sys.argv[5]) if len(sys.argv)>5 else 0
    ds=load_dataset("corypaik/musan","music",split="train",streaming=True).cast_column("audio",Audio(decode=False))
    for i,ex in enumerate(ds):
        if i<start or i%k!=k//2: continue
        if tot>=cap*3600: break
        y=dec(ex); y=y[:int(600*16000)]   # cap a track at 10 min
        sf.write(f"{outd}/music_{i:04d}.wav",y,16000,subtype="PCM_16"); tot+=len(y)/16000
        json.dump({"src":ex["path"],"source":ex["source"]},open(f"{outd}/music_{i:04d}.json","w")); print("music",i,ex["source"],round(len(y)/60,1),flush=True)
else:
    if which=="noise": ds=load_dataset("corypaik/musan","noise",split="train",streaming=True); k=1; cap=float(sys.argv[3]); tag="noise"
    else: ds=load_dataset("ashraq/esc50",split="train",streaming=True); k=4; cap=float(sys.argv[3]); tag="esc"
    ds=ds.cast_column("audio",Audio(decode=False))
    buf=[];n=0;tot=0;files=0;meta=[]
    for i,ex in enumerate(ds):
        if i%k!=0: continue
        if tot>=cap*3600: break
        y=dec(ex); buf.append(y); tot+=len(y)/16000; n+=len(y)
        meta.append(ex.get("path") or ex.get("category"))
        if n>=600*16000:
            sf.write(f"{outd}/{tag}_{files:02d}.wav",np.concatenate(buf),16000,subtype="PCM_16"); json.dump(meta,open(f"{outd}/{tag}_{files:02d}.json","w")); files+=1;buf=[];n=0;meta=[];print(tag,files,round(tot/3600,2),flush=True)
    if buf: sf.write(f"{outd}/{tag}_{files:02d}.wav",np.concatenate(buf),16000,subtype="PCM_16"); json.dump(meta,open(f"{outd}/{tag}_{files:02d}.json","w"))
