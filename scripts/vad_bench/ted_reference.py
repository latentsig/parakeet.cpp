#!/usr/bin/env python3
"""usage: ted_reference.py DATA_DIR TDT_0.6B_V3_F16.gguf
Word timestamps of each talk from the ASR model (30 s clips, batches of 8); compare_ted.py merges
words closer than 0.4 s into speech spans. The reference is therefore coarse and ASR-derived."""
import json,glob,sys,soundfile as sf,numpy as np
from pk import PK
pk=PK(sys.argv[2])
for f in sorted(glob.glob(sys.argv[1]+"/ted_*.wav")):
    y,_=sf.read(f,dtype="float32")
    if len(y)<16000*60: continue
    W=[];C=30*16000
    clips=[y[i:i+C] for i in range(0,len(y),C)]
    for b in range(0,len(clips),8):
        res=pk.words_batch(clips[b:b+8])
        for k,r in enumerate(res):
            off=(b+k)*30.0
            for w in r["words"]: W.append([w["start"]+off,w["end"]+off])
    json.dump(W,open(f.replace(".wav","_words.json"),"w"))
    print(f,len(y)/16000,len(W),flush=True)
