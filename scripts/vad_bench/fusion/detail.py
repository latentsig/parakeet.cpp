# Boundary error / missed segments / TED agreement for the cross-fitted systems
import json,numpy as np,soundfile as sf,subprocess
import run
from run import *
from analyze import crossfit,static,lrget,HEADS
folds=np.array([d["fold"] for d in D]); sp_idx=[i for i,d in enumerate(D) if not d["noise_only"]]
def build_systems(h):
    S={}
    for nm,(r,p) in {"Silero .5":("silero",(0.5,)),"head .5":("head",(0.5,)),"OR .5/.5":("or",(0.5,0.5)),"two-stage .5/.5/160/160/300":("two",(0.5,0.5,16,16,30))}.items():
        S[nm]=lambda d,f,r=r,p=p:raw_mask(r,p,d,h)
    for r,nm in (("silero","Silero tuned"),("head","head tuned"),("or","OR tuned"),("and","AND tuned"),("mean","mean tuned"),("two","two-stage tuned")):
        _,ch=crossfit(static(r,h),"f1"); S[nm]=lambda d,f,r=r,ch=ch:raw_mask(r,ch[f],d,h)
    models={}
    for kind in("lr6","hgb6","lr6nz","hgb6nz"):
        base=kind.replace("nz",""); _,ch=crossfit(lrget(h,kind),"f1")
        for f in range(4):
            tr=[i for i in range(len(D)) if folds[i]!=f and (kind.endswith("nz") or not D[i]["noise_only"])]
            X=np.concatenate([feats(D[i],h,base)[::10] for i in tr]); y=np.concatenate([D[i]["ref_mask"][::10] for i in tr])
            mdl=HistGradientBoostingClassifier(max_iter=120,learning_rate=0.1,random_state=0) if base=="hgb6" else LogisticRegression(C=1.0,max_iter=300)
            models[(kind,f)]=mdl.fit(X,y)
        S[f"{kind} tuned"]=lambda d,f,kind=kind,base=base,ch=ch:models[(kind,f)].predict_proba(feats(d,h,base))[:,1]>=ch[f][0]
    return S
def metrics(fn,h):
    st=[];en=[];miss=0;nref=0;nseg=0;cnt=np.zeros(4)
    for i in sp_idx:
        d=D[i]; m,regs=post(fn(d,d["fold"]),**POST_HEAD); a,b,ms=bnd(regs,d["ref"]); st+=a;en+=b;miss+=ms;nref+=len(d["ref"]);nseg+=len(regs); cnt+=counts(m,d["ref_mask"])
    q=lambda x:(np.median(x),np.percentile(x,90))
    return dict(start=q(st),end=q(en),miss=100*miss/nref,regions_per_ref=nseg/nref,PRF=[100*x for x in prf(cnt)])
out={}
for h in HEADS:
    out[h]={}
    for nm,fn in build_systems(h).items():
        out[h][nm]=metrics(fn,h); o=out[h][nm]
        print(f"{h:5s} {nm:30s} start {o['start'][0]:.0f}/{o['start'][1]:.0f} end {o['end'][0]:.0f}/{o['end'][1]:.0f} miss {o['miss']:.2f}% regions/ref {o['regions_per_ref']:.2f} PRF {o['PRF'][0]:.1f}/{o['PRF'][1]:.1f}/{o['PRF'][2]:.1f}",flush=True)
json.dump(out,open("detail.json","w"),default=float)
