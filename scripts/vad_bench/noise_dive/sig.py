import numpy as np
SR=16000
def rms(x): return float(np.sqrt(np.mean(np.square(x,dtype=np.float64))))
def at_db(x,db):  # set RMS to db dBFS
    return (x/ (rms(x)+1e-12) * 10**(db/20)).astype(np.float32)
def white(n,r): return r.standard_normal(n)
def colored(n,r,alpha):  # 1/f^alpha power spectrum  (alpha=1 pink, 2 brown)
    X=np.fft.rfft(r.standard_normal(n)); f=np.arange(len(X),dtype=float); f[0]=1
    y=np.fft.irfft(X/f**(alpha/2),n); return y
def shaped(n,r,spec):  # noise with a given long-term magnitude spectrum (len n//2+1 interpolated)
    X=np.fft.rfft(r.standard_normal(n)); f=np.linspace(0,1,len(X)); sp=np.interp(f,np.linspace(0,1,len(spec)),spec)
    return np.fft.irfft(X*sp,n)
def ltass(utts,nfft=1024):
    acc=np.zeros(nfft//2+1)
    for u in utts:
        for i in range(0,len(u)-nfft,nfft//2):
            acc+=np.abs(np.fft.rfft(u[i:i+nfft]*np.hanning(nfft)))**2
    return np.sqrt(acc/acc.sum())
def sine(n,f,db=-20): t=np.arange(n)/SR; return at_db(np.sin(2*np.pi*f*t),db)
def sweep(n,f0=50,f1=7500,db=-20):
    t=np.arange(n)/SR; T=n/SR; ph=2*np.pi*(f0*t+(f1-f0)*t**2/(2*T)); return at_db(np.sin(ph),db)
def clicks(n,r,rate=3.0,db=-12,decay_ms=8):  # claps/coughs-like: decaying noise bursts at Poisson times
    y=np.zeros(n); L=int(SR*decay_ms/1000*6)
    env=np.exp(-np.arange(L)/(SR*decay_ms/1000))
    for t in np.cumsum(r.exponential(1/rate,int(n/SR*rate*2))):
        i=int(t*SR)
        if i+L>=n: break
        y[i:i+L]+=r.standard_normal(L)*env
    return at_db(y,db)*1.0
def hum(n,f=50,db=-25):
    t=np.arange(n)/SR; y=sum(np.sin(2*np.pi*f*k*t)/k for k in (1,2,3,4,5)); return at_db(y,db)
def music(n,r,db=-20):  # chords of harmonic tones, changing every 0.5 s
    y=np.zeros(n); seg=SR//2; roots=[220,247,262,294,330,349,392]
    for i in range(0,n,seg):
        f0=r.choice(roots)*r.choice([0.5,1,2]); L=min(seg,n-i); t=np.arange(L)/SR
        v=sum(np.sin(2*np.pi*f0*m*t)/m for m in (1,2,3,4,5)) + sum(np.sin(2*np.pi*f0*1.26*m*t)/m for m in (1,2,3))
        env=np.minimum(1,np.minimum(t/0.02,(L/SR-t)/0.05)); y[i:i+L]=v*env
    return at_db(y,db)
