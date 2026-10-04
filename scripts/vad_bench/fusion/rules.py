import numpy as np
from fl import *
THR=[round(x,2) for x in np.arange(0.1,0.91,0.05)]
THR9=[round(x,1) for x in np.arange(0.1,0.91,0.1)]
def dilate(m,pre,post):
    """grow each run: `pre` cells earlier, `post` cells later"""
    if pre==0 and post==0: return m
    s,e=runs(m); out=m.copy(); n=len(m)
    for a,b in zip(s,e): out[max(0,a-pre):a]=True; out[b:min(n,b+post)]=True
    return out
def two_stage(S,H,pre,post,Y):
    """Silero decides; head extends boundaries by up to pre/post cells (only where head says speech, contiguous to the silero run)
    and fills silero gaps shorter than Y cells when the head calls >=50% of the gap speech."""
    out=S.copy(); n=len(S)
    if pre or post:
        s,e=runs(S)
        for a,b in zip(s,e):
            # walk outward while head is speech, max pre/post cells
            i=a
            while i>0 and a-i<pre and H[i-1]: i-=1
            out[i:a]=True
            j=b
            while j<n and j-b<post and H[j]: j+=1
            out[b:j]=True
    if Y:
        s,e=runs(~out)
        for a,b in zip(s,e):
            if a>0 and b<n and b-a<Y and H[a:b].mean()>=0.5: out[a:b]=True
    return out
def raw_mask(rule,p,d,h):
    s=d["silero"];hh=d[h]
    if rule=="silero": return s>=p[0]
    if rule=="head": return hh>=p[0]
    if rule=="or": return (s>=p[0])|(hh>=p[1])
    if rule=="and": return (s>=p[0])&(hh>=p[1])
    if rule=="mean": w,t=p; return (w*s+(1-w)*hh)>=t
    if rule=="max": return np.maximum(s,hh)>=p[0]
    if rule=="min": return np.minimum(s,hh)>=p[0]
    if rule=="two": ts,th,pre,post,Y=p; return two_stage(s>=ts,hh>=th,pre,post,Y)
    if rule=="switch": ts,th,T=p; return (hh>=th) if d["snr_est"]<T else (s>=ts)
    if rule=="oracle": ts,th,T=p; return (hh>=th) if d["snr_true"]<=T else (s>=ts)
    raise ValueError(rule)
def candidates(rule):
    if rule in("silero","head","max","min"): return [(t,) for t in THR]
    if rule in("or","and"): return [(a,b) for a in THR9 for b in THR9]
    if rule=="mean": return [(w,t) for w in(0.25,0.5,0.75) for t in THR]
    if rule=="two": return [(ts,th,pre,post,Y) for ts in(.3,.5,.7) for th in(.3,.5,.7) for pre in(0,8,16,32,48) for post in(0,8,16,32) for Y in(0,15,30,60)]
    if rule=="switch": return [(ts,th,T) for ts in(.3,.5,.7) for th in(.3,.5,.7) for T in(8,12,16,20,24,28,32,40)]
    if rule=="oracle": return [(ts,th,T) for ts in(.3,.5,.7) for th in(.3,.5,.7) for T in(5,10)]
