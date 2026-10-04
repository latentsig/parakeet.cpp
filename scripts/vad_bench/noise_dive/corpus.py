import os
FUSION=os.environ.get('FUSION_DIR','../fusion')  # scripts/vad_bench/fusion with data/ built by prep_libri.py
"""Rebuild fusion/ clips in memory with the same generators and seeds (fusion/lib.py is imported read-only)."""
import sys,json,numpy as np
sys.path.insert(0,FUSION)
import lib as FL
libri=list(np.load(FUSION+"/data/libri.npy",allow_pickle=True)); spk=np.load(FUSION+"/data/spk.npy")
def folds():
    rng=np.random.default_rng(11); sp_ids=rng.permutation(sorted(set(spk))); return {s:i%4 for i,s in enumerate(sp_ids)}
def speech_sets():
    fold_of=folds(); rng=np.random.default_rng(11); rng.permutation(sorted(set(spk)))
    sets=[]
    for f in range(4):
        idx=[i for i in range(len(libri)) if fold_of[spk[i]]==f]; idx=list(rng.permutation(idx))
        for k in range(min(10,len(idx)//5)): sets.append((f,[libri[i] for i in idx[k*5:(k+1)*5]]))
    return sets
def clean_clips():
    out=[]
    for k,(f,us) in enumerate(speech_sets()):
        y,spans=FL.build(us,np.random.default_rng(1000+k)); out.append((y,spans))
    return out
def gap_clips():
    """12 per (white|pink) x (20,10,5,0 dB): 2 utts, 30 s noise stretch, 3 utts; noise over the whole clip at the SNR (speech power)."""
    fold_of=folds(); meta=[]
    for ci,(kind,snr) in enumerate([(k,s) for k in("white","pink") for s in(20,10,5,0)]):
        for k in range(12):
            f=k%4; idx=[i for i in range(len(libri)) if fold_of[spk[i]]==f]; r=np.random.default_rng(300+ci*50+k); idx=list(r.permutation(idx))[:5]
            us=[libri[i] for i in idx]
            y1,s1=FL.build(us[:2],np.random.default_rng(400+k)); y2,s2=FL.build(us[2:],np.random.default_rng(500+k))
            off=len(y1)/16000+30.0
            y=np.concatenate([y1,np.random.default_rng(600+k).standard_normal(30*16000)*1e-3,y2]).astype(np.float32)
            spans=s1+[(a+off,b+off) for a,b in s2]
            clean=y.copy()
            y=FL.add_noise(y,spans,kind,snr,np.random.default_rng(7000+ci*100+k))
            meta.append(dict(name=f"gap-{kind}{snr}_{k:02d}",kind=kind,snr=snr,k=k,y=y,clean=clean,spans=spans,stretch=(s1[-1][1]+0.5,s2[0][0]+off-0.5)))
    return meta
