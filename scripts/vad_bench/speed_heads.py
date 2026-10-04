#!/usr/bin/env python3
"""usage: PARAKEET_LIB=... taskset -c CPUS speed_heads.py TALK.wav GGUF_DIR
VAD speed on the first 300 s: Silero ONNX on 1 thread vs the heads (8 threads), 4 reps."""
import sys,time,subprocess,numpy as np,soundfile as sf,torch
from heads_lib import silero,parakeet
from pk import PK
y=sf.read(sys.argv[1],dtype="float32")[0][:16000*300]   # 300 s
D=len(y)/16000
pks={"ultra_q8":PK(sys.argv[2]+"/ultra-q8_0.gguf"),"ultra_f16":PK(sys.argv[2]+"/ultra-f16.gguf"),"redux_packed":PK(sys.argv[2]+"/redux-keep.gguf"),"redux_deq_f16":PK(sys.argv[2]+"/redux-deq-f16.gguf")}
fns={"silero_onnx_1thr":lambda:silero(y,True)}
for n,p in pks.items(): fns[n+"_8thr"]=(lambda p=p:parakeet(p,y))
T={k:[] for k in fns}; loads=[]
for rep in range(4):
    loads.append(subprocess.getoutput("cut -d' ' -f1 /proc/loadavg"))
    for k,f in fns.items():
        t0=time.perf_counter();f();T[k].append(time.perf_counter()-t0)
print("load1 at each rep start:",loads)
for k,v in T.items(): print("%-18s best %.2fs  median %.2fs  -> %.0fx / %.0fx real time"%(k,min(v),np.median(v),D/min(v),D/np.median(v)))
