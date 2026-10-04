# False alarms inside a 30 s speech-free noise stretch embedded in a file that also has speech
import json,numpy as np,soundfile as sf
import run
from run import *
from analyze import crossfit,static,lrget,HEADS
G=json.load(open("gap.json")); PG=np.load("probs_gap.npz"); GD=[]
for m in G:
    n=int(round(m["dur"]/GR)); d=dict(m); d["n"]=n; d["ref_mask"]=np.zeros(n,bool)
    for s,e in m["ref"]: d["ref_mask"][int(round(s/GR)):int(round(e/GR))]=True
    for k in FS: d[k]=hold(PG[f"{m['name']}|{k}"],FS[k],n)
    y=sf.read(f"clips/{m['name']}.wav",dtype="float32")[0]; fr=len(y)//160; e2=(y[:fr*160].reshape(fr,160)**2).mean(1)+1e-12
    d["snr_est"]=10*np.log10(np.percentile(e2,90)/np.percentile(e2,10)); d["snr_true"]=int(m["cond"].lstrip("whitepink")); GD.append(d)
folds=np.array([d["fold"] for d in D]); easy=None
def fa_stretch(mask,d):
    m,regs=post(mask,**POST_HEAD); a,b=d["stretch"]; ia,ib=int(a/GR),int(b/GR)
    fp=m[ia:ib].mean(); nseg=sum(1 for s,e in regs if a<=s<b)   # regions that start inside the stretch
    return fp,nseg
out={}
for h in HEADS:
    sysmask={}
    # fixed systems
    for nm,(r,p) in {"Silero .5":("silero",(0.5,)),"head .5":("head",(0.5,)),"OR .5/.5":("or",(0.5,0.5)),"AND .5/.5":("and",(0.5,0.5)),"mean .5/.5":("mean",(0.5,0.5)),"max .5":("max",(0.5,)),"min .5":("min",(0.5,)),"two-stage .5/.5/160/160/300":("two",(0.5,0.5,16,16,30)),"noise-switch .5/.5/T12":("switch",(0.5,0.5,12))}.items():
        sysmask[nm]=lambda d,f,r=r,p=p:raw_mask(r,p,d,h)
    # CV-tuned parameters per fold (chosen on the other folds, as in the main tables)
    for r,nm in (("silero","Silero tuned"),("head","head tuned"),("or","OR tuned"),("and","AND tuned"),("max","max tuned"),("min","min tuned"),("mean","mean tuned"),("two","two-stage tuned"),("switch","noise-switch tuned")):
        _,ch=crossfit(static(r,h),"f1"); sysmask[nm]=lambda d,f,r=r,ch=ch:raw_mask(r,ch[f],d,h)
    # logistic regression / boosting, fitted on the other folds; thresholds 0.5 and CV-tuned
    models={}
    for kind in("lr2","lr6","hgb6","lr6nz","hgb6nz"):
        base=kind.replace("nz","")
        _,ch=crossfit(lrget(h,kind),"f1")
        for f in range(4):
            tr=[i for i in range(len(D)) if folds[i]!=f and (kind.endswith("nz") or not D[i]["noise_only"])]
            X=np.concatenate([feats(D[i],h,base)[::10] for i in tr]); y=np.concatenate([D[i]["ref_mask"][::10] for i in tr])
            mdl=HistGradientBoostingClassifier(max_iter=120,learning_rate=0.1,random_state=0) if base=="hgb6" else LogisticRegression(C=1.0,max_iter=300)
            models[(kind,f)]=mdl.fit(X,y)
        for thr,lab in ((0.5,"@.5"),(None,"tuned")):
            sysmask[f"{kind} {lab}"]=lambda d,f,kind=kind,base=base,thr=thr,ch=ch:models[(kind,f)].predict_proba(feats(d,h,base))[:,1]>=(thr if thr is not None else ch[f][0])
    res={}
    for nm,fn in sysmask.items():
        L={};
        for d in GD:
            fp,ns=fa_stretch(fn(d,d["fold"]),d); L.setdefault(d["snr_true"],[]).append((fp,ns))
        res[nm]={snr_:(float(np.mean([x[0] for x in v])),float(np.sum([x[1] for x in v]))) for snr_,v in L.items()}
        allv=[x for v in L.values() for x in v]; res[nm]["all"]=(float(np.mean([x[0] for x in allv])),float(np.sum([x[1] for x in allv])))
    out[h]=res
json.dump(out,open("gap_results.json","w"))
for h,res in out.items():
    print(f"## {h}: embedded 30 s noise stretch ({len(GD)} clips = 12 per level x2 noise types... see below), FP frame rate in the stretch % / false regions started in stretch (total over clips)")
    print("| system | 20 dB | 10 dB | 5 dB | 0 dB | all |"); print("|---|---|---|---|---|---|")
    for nm,r in res.items(): print(f"| {nm} | "+" | ".join(f"{100*r[k][0]:.1f}% / {r[k][1]:.0f}" for k in (20,10,5,0,"all"))+" |")
