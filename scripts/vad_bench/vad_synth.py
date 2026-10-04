import numpy as np
GR=0.01  # mask grid 10 ms
def mask(spans,n_sec):
    m=np.zeros(int(round(n_sec/GR)),bool)
    for s,e in spans:
        m[int(round(s/GR)):int(round(e/GR))]=True
    return m
def trim(u):
    fr=len(u)//160; r=np.sqrt((u[:fr*160].reshape(fr,160)**2).mean(1))
    idx=np.where(r>r.max()*0.01)[0]
    return u[idx[0]*160:(idx[-1]+1)*160]
def pink(n,rng):
    X=np.fft.rfft(rng.standard_normal(n)); f=np.arange(len(X)); f[0]=1
    y=np.fft.irfft(X/np.sqrt(f),n); return y/y.std()
def build(utts,rng):
    g=lambda a,b: rng.uniform(a,b)
    parts=[];spans=[];t=g(0.5,1.5)
    parts.append(rng.standard_normal(int(t*16000))*1e-3)
    for i,u in enumerate(utts):
        u=trim(u); spans.append((t,t+len(u)/16000)); parts.append(u); t+=len(u)/16000
        gap=g(0.3,2.0) if i<len(utts)-1 else g(0.5,1.5)
        parts.append(rng.standard_normal(int(gap*16000))*1e-3); t+=int(gap*16000)/16000
    return np.concatenate(parts).astype(np.float32),spans
def add_noise(y,spans,kind,snr,rng):
    if kind=="clean": return y
    sp=np.concatenate([y[int(s*16000):int(e*16000)] for s,e in spans]); P=(sp**2).mean()
    n=rng.standard_normal(len(y)) if kind=="white" else pink(len(y),rng)
    n=n*np.sqrt(P/10**(snr/10))
    return np.clip(y+n,-1,1).astype(np.float32)
def counts(pred,ref):
    n=min(len(pred),len(ref)); p=pred[:n];r=ref[:n]
    return np.array([(p&r).sum(),(p&~r).sum(),(~p&r).sum(),(~p&~r).sum()])
def prf(c):
    tp,fp,fn,tn=c; P=tp/max(1,tp+fp);R=tp/max(1,tp+fn);return P,R,2*P*R/max(1e-9,P+R)
def kappa(c):
    tp,fp,fn,tn=c;n=c.sum();po=(tp+tn)/n;pe=((tp+fp)*(tp+fn)+(fn+tn)*(fp+tn))/n**2;return (po-pe)/(1-pe)
def bound_err(pred,ref,idx):
    out=[];miss=0
    pb=np.array([p[idx] for p in pred]) if pred else np.array([])
    for r in ref:
        if len(pb)==0: miss+=1;continue
        d=np.abs(pb-r[idx]).min()
        if d>1.0: miss+=1
        else: out.append(d*1000)
    return out,miss

