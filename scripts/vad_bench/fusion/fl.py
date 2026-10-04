"""Fusion library: 10 ms grid, segmenter semantics, metrics."""
import json,numpy as np,soundfile as sf
GR=0.01
FS={"silero":0.032,"ultra":0.08,"redux":0.08}
def hold(p,fs,n):
    """sample-and-hold: grid cell j (center (j+.5)*10ms) takes the value of the frame containing it; cells past the last frame get 0."""
    idx=((np.arange(n)+0.5)*GR/fs).astype(int)
    out=np.zeros(n,np.float32); ok=idx<len(p); out[ok]=p[idx[ok]]; return out
def runs(m):
    """[start,end) of True runs"""
    d=np.diff(np.concatenate([[0],m.astype(np.int8),[0]]))
    return np.flatnonzero(d==1),np.flatnonzero(d==-1)
def post(m,bridge=0.1,min_speech=0.1,min_pause=0.2,pad=0.0):
    """Repo segmenter semantics (vad_segmenter.cpp speech_regions) on the 10 ms grid: bridge short speech gaps,
    drop short runs, merge regions closer than min_pause (the gap becomes speech), then pad."""
    n=len(m); m=m.copy()
    s,e=runs(~m)
    for a,b in zip(s,e):
        if a>0 and b<n and (b-a)*GR+1e-9<bridge: m[a:b]=True
    s,e=runs(m)
    for a,b in zip(s,e):
        if (b-a)*GR+1e-9<min_speech: m[a:b]=False
    s,e=runs(m); pf=max(1,int(np.ceil(min_pause/GR-1e-9)))
    out=np.zeros(n,bool); regs=[]
    for a,b in zip(s,e):
        if regs and a-regs[-1][1]<pf: regs[-1][1]=b
        else: regs.append([a,b])
    pp=int(round(pad/GR))
    for a,b in regs: out[max(0,a-pp):min(n,b+pp)]=True
    return out,[(a*GR,b*GR) for a,b in regs]
POST_HEAD=dict(bridge=0.1,min_speech=0.1,min_pause=0.2,pad=0.0)       # unified post (= head defaults)
POST_SIL=dict(bridge=0.1,min_speech=0.25,min_pause=0.1,pad=0.03)      # Silero native defaults
def counts(p,r): return np.array([(p&r).sum(),(p&~r).sum(),(~p&r).sum(),(~p&~r).sum()],np.int64)
def prf(c):
    tp,fp,fn,_=c; P=tp/max(1,tp+fp);R=tp/max(1,tp+fn);return P,R,2*P*R/max(1e-12,P+R)
def bnd(regs,ref):
    """nearest predicted boundary distance (ms) per reference boundary; miss = no boundary within 1 s"""
    st=[];en=[];miss=0
    pb=np.array([x for r in regs for x in r])
    for rs,re_ in ref:
        if len(pb)==0: miss+=1;continue
        ds=np.abs(np.array([r[0] for r in regs])-rs).min();de=np.abs(np.array([r[1] for r in regs])-re_).min()
        if min(ds,de)>1.0: miss+=1  # same convention as the earlier comparison: miss if a boundary is >1 s away
        if ds<=1.0: st.append(ds*1000)
        if de<=1.0: en.append(de*1000)
    return st,en,miss
def load(name_filter=None):
    meta=json.load(open("clips.json")); P=np.load("probs.npz")
    D=[]
    for m in meta:
        n=int(round(m["dur"]/GR)); d=dict(m); d["n"]=n
        d["ref_mask"]=np.zeros(n,bool)
        for s,e in m["ref"]: d["ref_mask"][int(round(s/GR)):int(round(e/GR))]=True
        for k in FS: d[k]=hold(P[f"{m['name']}|{k}"],FS[k],n)
        y=sf.read(f"clips/{m['name']}.wav",dtype="float32")[0]
        fr=len(y)//160; e2=(y[:fr*160].reshape(fr,160)**2).mean(1)+1e-12
        d["snr_est"]=10*np.log10(np.percentile(e2,90)/np.percentile(e2,10))
        D.append(d)
    return D
