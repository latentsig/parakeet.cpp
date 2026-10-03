#!/usr/bin/env python3
"""usage: fetch_ted_talks.py OUT_DIR [N_TALKS]  -> OUT_DIR/ted_<name>.wav, the first N talks under 1500 s of
distil-whisper/tedlium-long-form (test). The published run used three talks; which ones is not recorded."""
import io, re, sys, numpy as np, soundfile as sf, librosa
from datasets import load_dataset, Audio
D=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 3
# TED-LIUM long-form: first talks
ds=load_dataset("distil-whisper/tedlium-long-form",split="test",streaming=True).cast_column("audio",Audio(decode=False))
n=0
for i,ex in enumerate(ds):
    y,sr=sf.read(io.BytesIO(ex["audio"]["bytes"]),dtype="float32")
    if y.ndim>1:y=y.mean(1)
    if sr!=16000:y=librosa.resample(y,orig_sr=sr,target_sr=16000)
    if len(y)/16000>1500: continue
    name=re.sub(r"[^A-Za-z0-9_-]","",(ex["audio"]["path"] or f"talk{i}").split("/")[-1].rsplit(".",1)[0]) or f"talk{i}"
    sf.write(f"{D}/ted_{name}.wav",y,16000,subtype="PCM_16")
    print(name,len(y)/16000,flush=True); n+=1
    if n>=N: break
