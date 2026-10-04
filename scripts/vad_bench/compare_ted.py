#!/usr/bin/env python3
"""usage: compare_ted.py DATA_DIR GGUF_DIR OUT.json
DATA_DIR holds ted_*.wav and the matching ted_*_words.json from ted_reference.py.
The speed it prints is NOT a valid benchmark (other load); use speed_heads.py."""
import json,glob,sys,time,subprocess,numpy as np,soundfile as sf
from heads_lib import *
from pk import PK
DATA,G,OUT=sys.argv[1],sys.argv[2],sys.argv[3]
print("uptime before:",subprocess.getoutput("uptime"))
talks=[f for f in sorted(glob.glob(DATA+"/ted_*.wav")) if "S103" not in f]   # the original run left out one talk with S103 in its name; reason not recorded
ys={f:sf.read(f,dtype="float32")[0] for f in talks}
refs={}
for f in talks:
    W=json.load(open(f.replace(".wav","_words.json")))
    spans=[];
    for s,e in W:
        if spans and s-spans[-1][1]<0.4: spans[-1][1]=max(spans[-1][1],e)
        else: spans.append([s,e])
    refs[f]=spans
res={};preds={}
def run(name,fn):
    cs=np.zeros(4);tt=0;ds=0;
    for f in talks:
        y=ys[f];t0=time.perf_counter();p=fn(y);dt=time.perf_counter()-t0
        preds[(name,f)]=p;tt+=dt;ds+=len(y)/16000
        L=len(y)/16000+1;cs+=counts(mask(p,L),mask(refs[f],L))
    P,R,F=prf(cs);res[name]=dict(P=P,R=R,F1=F,rtf_x=ds/tt,sec=tt,audio=ds,speech_frac=None)
    print(name,"P %.1f R %.1f F1 %.1f  VAD speed %.0fx real time (%.1f s for %.0f s)"%(P*100,R*100,F*100,ds/tt,tt,ds),flush=True)
run("silero_matched",lambda y:silero(y,True))
run("silero_default",lambda y:silero(y,False))
for n,f in(("ultra_q8","ultra-q8_0.gguf"),("redux_packed","redux-keep.gguf")):
    pk=PK(G+"/"+f); run(n,lambda y:parakeet(pk,y)); 
    out=pk.vad(ys[talks[1]],{"threshold":0.5})
    del pk
# reference speech fraction and agreement
cs=np.zeros(4)
for f in talks:
    L=len(ys[f])/16000+1; r=mask(refs[f],L); cs+=np.array([r.sum(),0,0,len(r)-r.sum()])
print("ref speech fraction %.1f%%"%(100*cs[0]/(cs[0]+cs[3])))
for n in ["silero_matched","silero_default","ultra_q8","redux_packed"]:
    t=sum(mask(preds[(n,f)],len(ys[f])/16000+1).sum() for f in talks);a=sum(len(mask([],len(ys[f])/16000+1)) for f in talks)
    print(n,"speech fraction %.1f%%"%(100*t/a))
print("AGREEMENT on TED talks")
for a,b in(("ultra_q8","silero_matched"),("redux_packed","silero_matched"),("ultra_q8","redux_packed")):
    c=np.zeros(4)
    for f in talks:
        L=len(ys[f])/16000+1;c+=counts(mask(preds[(a,f)],L),mask(preds[(b,f)],L))
    print(a,"vs",b,"F1 %.1f kappa %.3f"%(prf(c)[2]*100,kappa(c)))
print("talks",[(f,round(len(ys[f])/16000)) for f in talks])
json.dump(res,open(OUT,"w"),indent=1)
print("uptime after:",subprocess.getoutput("uptime"))
