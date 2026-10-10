import json,numpy as np,vr,sys
D=vr.load_recordings(); bad=0;tot=0;maxd=0
OPT={"silero":(0.25,0.1,0.03),"ultra":(0.1,0.2,0.0),"redux":(0.1,0.2,0.0)}
for m in D:
    P=vr.load_probs(m["id"])
    for k in vr.FS:
        j=json.load(open(f"{vr.ROOT}/probs/{m['id']}.{k}.json"))
        ms,mp,pad=OPT[k]
        _,st,en=vr.native_mask(P[k],vr.FS[k],j["duration"],0.5,0.1,ms,mp,pad,int(round(m["dur"]/vr.GR)))
        cli=j["segments"]; tot+=1
        ok=len(cli)==len(st) and all(abs(c["start"]-a)<=0.0006 and abs(c["end"]-b)<=0.0006 for c,a,b in zip(cli,st,en))
        if not ok: bad+=1; print("MISMATCH",m["id"],k,len(cli),len(st))
print("checked",tot,"mismatch",bad)
