import json,os,sys,numpy as np
from vad_synth import *
SB=sys.argv[1]   # DATA_DIR, as for silero_collect.py; run in the directory that holds probs.npz
P=np.load("probs.npz"); S=json.load(open("segs_own_default.json"))
meta=json.load(open(SB+"/clips.json"))
SYS=["onnx","wcpp_conv623","wcpp_repo620","pk_f32","pk_f16"]
conds=["clean0","clean","white10","pink10"]
def post(p,thr=0.5,min_speech=0.1,min_sil=0.2,pad=0.0,fs=0.032,total=None):
    # silero get_speech_timestamps-style hysteresis (neg = thr-0.15), no max-speech split
    neg=max(thr-0.15,0.01); segs=[];cur=None;tmp=None
    for i,v in enumerate(p):
        t=i*fs
        if v>=thr and tmp is not None: tmp=None
        if v>=thr and cur is None: cur=t; continue
        if v<neg and cur is not None:
            if tmp is None: tmp=t
            if t-tmp<min_sil: continue
            if tmp-cur>min_speech: segs.append([cur,tmp])
            cur=None;tmp=None
    end=len(p)*fs
    if cur is not None and end-cur>min_speech: segs.append([cur,end])
    out=[]
    for i,(a,b) in enumerate(segs):
        out.append([max(0,a-pad),min(total or end,b+pad)])
    return out
def lens(a,b): n=min(len(a),len(b)); return a[:n],b[:n]
# ---- 1. probability parity (vs onnxruntime) ----
names=[m["name"] for m in meta]
def parity(sysn,ref="onnx",nms=names+["ted_talk"]):
    mx=0;sm=0;cnt=0;dec=0;d1=0;big=0
    for n in nms:
        a,b=lens(P[f"{n}|{sysn}"],P[f"{n}|{ref}"]); d=np.abs(a-b)
        mx=max(mx,d.max());sm+=d.sum();cnt+=len(d);dec+=((a>=.5)!=(b>=.5)).sum();big+=(d>0.05).sum()
    return mx,sm/cnt,100*(1-dec/cnt),dec,100*big/cnt
out=[]
out.append("PROBABILITY PARITY vs onnxruntime (v6.2.3 ONNX, official 64-sample context), all 120 clips + 600 s talk, per 32 ms chunk")
out.append("system | max abs diff | mean abs diff | frames > 0.05 off | decision agreement @0.5 | flipped frames")
for s in SYS[1:]:
    mx,mn,ag,fl,big=parity(s); out.append(f"{s} | {mx:.4f} | {mn:.5f} | {big:.2f}% | {ag:.3f}% | {fl}")
out.append("pairwise (whisper.cpp converted v6.2.3 vs parakeet f32):")
mx,mn,ag,fl,big=parity("wcpp_conv623","pk_f32");out.append(f"wcpp_conv623 vs pk_f32 | {mx:.4f} | {mn:.5f} | {big:.2f}% | {ag:.3f}% | {fl}")
mx,mn,ag,fl,big=parity("wcpp_conv623","wcpp_repo620");out.append(f"wcpp_conv623 vs wcpp_repo620 | {mx:.4f} | {mn:.5f} | {big:.2f}% | {ag:.3f}% | {fl}")
out.append("per-region parity vs onnx (mean abs diff) : first chunk of each clip / rest")
for s in SYS[1:]:
    f=np.mean([abs(P[f"{n}|{s}"][0]-P[f"{n}|onnx"][0]) for n in names]); r=np.mean([np.abs(*[lens(P[f"{n}|{s}"],P[f"{n}|onnx"])[0]-lens(P[f"{n}|{s}"],P[f"{n}|onnx"])[1]])[1:].mean() for n in names])
    out.append(f"{s} first-chunk {f:.5f} rest {r:.5f}")
# ---- 2. segment quality ----
def agg(getseg,cs):
    c=np.zeros(4);bs=[];be=[];miss=0;nb=0
    for m in meta:
        if m["cond"] not in cs: continue
        L=m["dur"]; pr=getseg(m["name"],L); ref=m["ref"]
        c+=counts(mask(pr,L),mask(ref,L))
        e1,m1=bound_err(pr,ref,0);e2,m2=bound_err(pr,ref,1);bs+=e1;be+=e2;miss+=m1+m2;nb+=2*len(ref)
    return c,bs,be,miss,nb
sysdefs={}
for s in ["onnx","wcpp_conv623","pk_f32","pk_f16"]:
    sysdefs[f"{s} [matched post]"]=(lambda s:lambda n,L:post(P[f"{n}|{s}"],total=L))(s)
sysdefs["wcpp_conv623 [whisper.cpp own default]"]=lambda n,L:S[n]["wcpp_conv623_seg_default"]
sysdefs["wcpp_conv623 [whisper.cpp C++ post, matched params]"]=lambda n,L:S[n]["wcpp_conv623_seg_matched"]
sysdefs["pk_f32 [parakeet own default]"]=lambda n,L:S[n]["pk_f32_seg_default"]
out.append("\nSEGMENT QUALITY, frame level on 10 ms grid vs synthetic reference (P/R/F1 %), 40 clips per condition")
out.append("matched post = thr 0.5 / neg 0.35 / min speech 100 ms / min silence 200 ms / pad 0 (same python code on every probability stream)")
allc=["clean","white10","pink10"]
out.append("system | "+" | ".join(allc)+" | all")
res={}
for k,g in sysdefs.items():
    cells=[]
    for cs in [[c] for c in allc]+[allc]:
        c,*_=agg(g,cs); p,r,f=prf(c); cells.append(f"{p*100:.1f}/{r*100:.1f}/{f*100:.1f}")
    out.append(f"{k} | "+" | ".join(cells))
out.append("\nBOUNDARY ERROR vs reference, all conditions pooled: start median/p90, end median/p90, miss%")
for k,g in sysdefs.items():
    c,bs,be,miss,nb=agg(g,allc)
    out.append(f"{k} | start {np.median(bs):.0f}/{np.percentile(bs,90):.0f} | end {np.median(be):.0f}/{np.percentile(be,90):.0f} | miss {100*miss/nb:.1f}%")
# ---- 4. agreement between implementations ----
out.append("\nAGREEMENT of speech masks (matched post), all 120 clips pooled: F1 / kappa / IoU")
pairs=[("wcpp_conv623","pk_f32"),("wcpp_conv623","onnx"),("pk_f32","onnx"),("pk_f16","pk_f32"),("wcpp_conv623","wcpp_repo620")]
def agree(a,b,nms,Ls):
    c=np.zeros(4)
    for n,L in zip(nms,Ls): c+=counts(mask(post(P[f"{n}|{a}"],total=L),L),mask(post(P[f"{n}|{b}"],total=L),L))
    return c
for a,b in pairs:
    c=agree(a,b,names,[m["dur"] for m in meta]); p,r,f=prf(c)
    out.append(f"{a} vs {b} | F1 {f*100:.2f} | kappa {kappa(c):.4f} | IoU {100*c[0]/(c[0]+c[1]+c[2]):.2f}")
out.append("\nAGREEMENT on the 600 s TED talk (no reference): F1 / kappa, speech fraction")
L=600.0
for s in ["onnx","wcpp_conv623","pk_f32"]:
    m=mask(post(P[f"ted_talk|{s}"],total=L),L); out.append(f"{s} speech fraction {100*m.mean():.1f}%")
for a,b in pairs[:3]:
    c=agree(a,b,["ted_talk"],[L]); p,r,f=prf(c); out.append(f"{a} vs {b} | F1 {f*100:.2f} | kappa {kappa(c):.4f}")
out.append("TED talk flipped frames @0.5 vs onnx: "+", ".join(f"{s} {int(((lens(P[f'ted_talk|{s}'],P['ted_talk|onnx'])[0]>=.5)!=(lens(P[f'ted_talk|{s}'],P['ted_talk|onnx'])[1]>=.5)).sum())}/{len(P['ted_talk|onnx'])}" for s in SYS[1:]))
open("results.txt","w").write("\n".join(out)); print("\n".join(out))
