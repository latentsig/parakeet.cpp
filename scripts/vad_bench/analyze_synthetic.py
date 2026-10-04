#!/usr/bin/env python3
"""usage: analyze_synthetic.py synthetic_raw.json  -> tables on stdout"""
import json,sys,numpy as np
from heads_lib import mask,counts,prf,kappa,bound_err
R=json.load(open(sys.argv[1]))
conds=R["conds"];refs=R["refs"];P=R["preds"]
dur=lambda i: refs[i][-1][1]+1.0  # unused
def total_sec(i): 
    return None
import soundfile
# clip length: need exact; recompute from the raw: last pred... use max(ref end, pred end)+2 s margin rounding => use mask length from ref end + 1.5 (trailing silence >=0.5)
def n_sec(i): return max(refs[i][-1][1]+0.5, max([e for s,e in (P['silero_matched'][i] or [(0,0)])]+[0]))+0.0
order=["clean","white20","white10","white5","white0","pink20","pink10","pink5","pink0"]
def agg(sysname,cs):
    c=np.zeros(4);bs=[];be=[];ms=0;me=0;nb=0
    for i,cn in enumerate(conds):
        if cn not in cs or P[sysname][i] is None: continue
        L=n_sec(i)+3
        ref=mask(refs[i],L);pr=mask(P[sysname][i],L)
        c+=counts(pr,ref)
        e1,m1=bound_err(P[sysname][i],refs[i],0);e2,m2=bound_err(P[sysname][i],refs[i],1)
        bs+=e1;be+=e2;ms+=m1;me+=m2;nb+=len(refs[i])
    return c,bs,be,ms,me,nb
out={}
systems=["silero_matched","silero_default","ultra_q8","redux_packed"]
print("FRAME-LEVEL (10 ms grid), thr 0.5")
print("cond | "+" | ".join(f"{s} P/R/F1" for s in systems))
rows=[]
for cn in order:
    cells=[]
    for s in systems:
        c,*_=agg(s,[cn]);p,r,f=prf(c);cells.append(f"{p*100:.1f}/{r*100:.1f}/{f*100:.1f}");out[f"{cn}|{s}"]=[p,r,f]
    print(cn," | ".join(cells))
print("\nBOUNDARY ERROR ms (median / p90 / miss%) start;end, pooled")
for cn in ["clean","white10","white0","pink10","pink0"]:
    cells=[]
    for s in systems:
        c,bs,be,ms,me,nb=agg(s,[cn])
        f=lambda a:(f"{np.median(a):.0f}/{np.percentile(a,90):.0f}" if a else "-")
        cells.append(f"S {f(bs)} ({100*ms/nb:.1f}%) E {f(be)} ({100*me/nb:.1f}%)")
    print(cn," | ".join(cells))
print("all-conditions boundary:")
for s in systems:
    c,bs,be,ms,me,nb=agg(s,order); a=bs+be
    print(s,f"median {np.median(a):.0f} p90 {np.percentile(a,90):.0f} start-med {np.median(bs):.0f} end-med {np.median(be):.0f} miss {100*(ms+me)/(2*nb):.1f}%")
print("\nTHRESHOLD SWEEP F1 (clean,white10,white0,pink10 pooled)  P/R/F1")
sub=["clean","white10","white0","pink10"]
for base in ["silero_matched","ultra_q8","redux_packed"]:
    cells=[]
    for t in ("@0.3","","@0.7"):
        nm=base+t; c,*_=agg(nm,sub); p,r,f=prf(c); cells.append(f"thr{t or '@0.5'} {p*100:.1f}/{r*100:.1f}/{f*100:.1f}")
    print(base," | ".join(cells))
print("\nAGREEMENT silero_matched vs parakeet (all synthetic): F1 of speech mask, kappa, IoU")
for s in ["ultra_q8","redux_packed"]:
    c=np.zeros(4)
    for i in range(len(conds)):
        L=n_sec(i)+3;c+=counts(mask(P[s][i],L),mask(P["silero_matched"][i],L))
    p,r,f=prf(c);print(s,f"F1 {f*100:.1f} kappa {kappa(c):.3f} IoU {c[0]/(c[0]+c[1]+c[2])*100:.1f}")
c=np.zeros(4)
for i in range(len(conds)):
    L=n_sec(i)+3;c+=counts(mask(P["ultra_q8"][i],L),mask(P["redux_packed"][i],L))
print("ultra vs redux",f"F1 {prf(c)[2]*100:.1f} kappa {kappa(c):.3f}")
