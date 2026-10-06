#!/usr/bin/env python3
"""Stream diarizers-community voxconverse / ami(sdm), keep every k-th recording up to a duration cap. Writes wav (16k mono int16) + turns json."""
import io, json, os, sys, re
import numpy as np, soundfile as sf, librosa
ROOT=os.environ.get("VAD_REAL_ROOT") or os.getcwd()
os.environ.setdefault("HF_HOME",f"{ROOT}/hfhome")
from datasets import Audio, load_dataset
name, cfg, splits, k, cap_h, outd = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
tot=0.0
for split in splits:
    ds = load_dataset(name, None if cfg=="-" else cfg, split=split, streaming=True).cast_column("audio", Audio(decode=False))
    for i, ex in enumerate(ds):
        if i % k != (k//2): continue
        if tot/3600 >= cap_h: break
        y, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        if y.ndim > 1: y = y.mean(axis=1)
        if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        rid=f"{split}_{i:04d}"
        sf.write(f"{outd}/{rid}.wav", y, 16000, subtype="PCM_16")
        json.dump({"id":rid,"split":split,"dur":len(y)/16000,"start":ex["timestamps_start"],"end":ex["timestamps_end"],"speakers":ex["speakers"]}, open(f"{outd}/{rid}.json","w"))
        tot+=len(y)/16000; print(rid, round(len(y)/60,1), "min; total h", round(tot/3600,2), flush=True)
