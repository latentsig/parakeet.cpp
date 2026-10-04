import numpy as np,json,torch
from vadlib import *; from sig import *; import corpus
from sklearn.linear_model import LinearRegression, Ridge
from scipy.signal import butter, sosfiltfilt
rng=np.random.default_rng(21); N=30*SR
lib=corpus.libri; spec=ltass(lib[:60])
def feats(x):
    """per-80ms-frame descriptors from the audio, on a +-0.24 s window (the head's span ~0.5 s)."""
    m=raw_logmel(x).double(); n=m.shape[0]-1; m=m[:n]
    mn=norm_mel(raw_logmel(x)).double()              # what the net sees (per-file normalised)
    T=n//8
    E=np.exp(m.numpy())                               # linear mel power
    mnn=mn.numpy(); mm=m.numpy()
    env=mm.mean(1)                                    # log-mel envelope (mean over bins)
    sos=butter(2,[2,8],btype='band',fs=100,output='sos'); mod=sosfiltfilt(sos,env)**2
    k=np.arange(128); tilt_f=lambda row: np.polyfit(k,row,1)[0]
    out=[]
    for t in range(T):
        a=max(0,8*t-8); b=min(n,8*t+16)               # +-8 mel frames around the 8-frame block (0.24 s each side)
        w=mnn[a:b]; e=E[a:b]
        flat=np.mean(np.exp(np.log(e+2**-24).mean(1))/(e.mean(1)+2**-24))
        d=np.diff(mnn[a:b],axis=0); flux=np.mean(np.maximum(d,0)) if len(d) else 0
        out.append([w.mean(),                           # energy (relative to file)
                    mm[8*t:8*t+8].mean(),                # absolute log energy of the block
                    flux, flat, mod[a:b].mean(), tilt_f(w.mean(0))])
    return np.array(out), mnn
names=['frame energy (norm.)','abs log-energy','spectral flux','spectral flatness','modulation 2-8 Hz','spectral tilt']
def collect(model,clips):
    X=[];Z=[]
    for x in clips:
        f,_=feats(x); z=ref(model).logits(x)
        n=min(len(f),len(z)); X.append(f[:n]); Z.append(z[:n])
    return np.concatenate(X),np.concatenate(Z)

from sklearn.ensemble import HistGradientBoostingRegressor
def winmel(x,T):
    mn=norm_mel(raw_logmel(x)).numpy(); n=len(mn); out=[]
    for t in range(T):
        a=max(0,8*t-8); b=min(n,8*t+16); out.append(mn[a:b].mean(0))
    return np.array(out)
kinds={'white':lambda: white(N,rng),'pink':lambda: colored(N,rng,1),'brown':lambda: colored(N,rng,2),'speech-shaped':lambda: shaped(N,rng,spec),'speech (LibriSpeech)':None}
res={}
for m in ['redux','ultra']:
    for kn,gen in kinds.items():
        clips=[at_db(gen(),-25) for _ in range(6)] if gen else [np.concatenate(lib[10+6*i:16+6*i]).astype(np.float32)[:N] for i in range(6)]
        X=[];Z=[];W=[]
        for x in clips:
            f,_=feats(x); z=ref(m).logits(x); n=min(len(f),len(z)); X.append(f[:n]); Z.append(z[:n]); W.append(winmel(x,n))
        X=np.concatenate(X);Z=np.concatenate(Z);W=np.concatenate(W); Zc=np.clip(Z,-8,8)
        Xs=(X-X.mean(0))/(X.std(0)+1e-9)
        # file-level split for validation: train on first 4 clips, test on last 2
        ntr=int(len(Z)*4/6)
        lr=LinearRegression().fit(Xs[:ntr],Zc[:ntr]); r2te=lr.score(Xs[ntr:],Zc[ntr:])
        corr=[np.corrcoef(Xs[:,j],Zc)[0,1] for j in range(6)]
        gb=HistGradientBoostingRegressor(max_iter=200).fit(Xs[:ntr],Zc[:ntr]); gbr2=gb.score(Xs[ntr:],Zc[ntr:])
        rd=Ridge(alpha=10).fit(W[:ntr],Zc[:ntr]); rdr2=rd.score(W[ntr:],Zc[ntr:])
        # energy-only
        e1=LinearRegression().fit(Xs[:ntr,:1],Zc[:ntr]).score(Xs[ntr:,:1],Zc[ntr:])
        dom=names[int(np.argmax(np.abs(lr.coef_)))]
        res[f'{m}|{kn}']=dict(speech=float((Z>=0).mean()),r2_lin6=r2te,r2_gbm6=gbr2,r2_spec128=rdr2,r2_energy_only=e1,corr=corr,coef=lr.coef_.tolist(),logit_mean=float(Z.mean()),logit_std=float(Z.std()))
        print(f'{m:6s} {kn:22s} speech {100*(Z>=0).mean():5.1f}%  logit mean {Z.mean():+.2f} sd {Z.std():.2f} | held-out R2: 6-feature linear {r2te:+.2f}, 6-feature GBM {gbr2:+.2f}, energy only {e1:+.2f}, 128-bin window-mel ridge {rdr2:+.2f} | corr '+' '.join(f'{c:+.2f}' for c in corr)+f' | dominant {dom}',flush=True)
print('features:',names)
json.dump(res,open('res_B4.json','w'),indent=1)
