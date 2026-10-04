import numpy as np,json
from vadlib import *; from sig import *; import corpus
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
models=['redux','ultra']; cl=corpus.clean_clips(); gaps=corpus.gap_clips()
rng=np.random.default_rng(5); N=30*SR
D={m:dict(speech_clean=[],pause_clean=[],speech_noisy={},stretch={},nz={}) for m in models}
for m in models:
    for y,spans in cl:
        z=logits_blocks(m,y); sp=frame_mask(spans,len(z))
        # pause frames: >=0.3 s from any speech (gap between utterances, digital -60 dBFS dither)
        t=(np.arange(len(z))+.5)*.08; far=np.ones(len(z),bool)
        for a,b in spans: far&=~((t>a-0.3)&(t<b+0.3))
        D[m]['speech_clean'].append(z[sp]); D[m]['pause_clean'].append(z[far])
    for g in gaps:
        z=logits_blocks(m,g['y']); n=len(z); t=(np.arange(n)+.5)*.08; st=(t>=g['stretch'][0])&(t<g['stretch'][1]); sp=frame_mask(g['spans'],n)
        k=g['snr']; D[m]['speech_noisy'].setdefault(k,[]).append(z[sp]); D[m]['stretch'].setdefault(k,[]).append(z[st])
    for lvl in [-50,-40,-30,-23,-15]:
        zs=[ref(m).logits(at_db(white(N,rng) if i%2==0 else colored(N,rng,1),lvl)) for i in range(6)]
        D[m]['nz'][lvl]=np.concatenate(zs)
def cat(l): return np.concatenate(l)
def best(zs,zn):
    y=np.r_[np.ones(len(zs)),np.zeros(len(zn))]; s=np.r_[zs,zn]; auc=roc_auc_score(y,s); fpr,tpr,th=roc_curve(y,s)
    j=np.argmax(tpr-fpr); P=tpr*len(zs)/(tpr*len(zs)+fpr*len(zn)+1e-9); F=2*P*tpr/(P+tpr+1e-9)
    return auc,th[j],fpr[j],tpr[j],th[np.argmax(F)]
out={}
print('| model | comparison | AUC | Youden-best logit thr (p) | TPR / FPR at best | F1-best logit thr (p) | speech/noise at default 0 |')
print('|---|---|---|---|---|---|---|')
for m in models:
    d=D[m]; zs=cat(d['speech_clean'])
    rows=[('speech (clean clips) vs pauses in the same clips',zs,cat(d['pause_clean']))]
    for k in [20,10,5,0]:
        rows.append((f'speech vs embedded noise stretch, SNR {k} dB',cat(d['speech_noisy'][k]),cat(d['stretch'][k])))
    rows.append(('speech (clean) vs noise-only files -50..-15 dBFS pooled',zs,cat(list(d['nz'].values()))))
    rows.append(('speech (clean) vs noise-only file at -23 dBFS (speech-level)',zs,d['nz'][-23]))
    for nm,a,b in rows:
        auc,thr,fpr,tpr,thF=best(a,b)
        print(f'| {m} | {nm} | {auc:.3f} | {thr:+.2f} ({sig(thr):.2f}) | {tpr:.2f} / {fpr:.2f} | {thF:+.2f} ({sig(thF):.2f}) | {100*(a>=0).mean():.0f}% / {100*(b>=0).mean():.0f}% |')
        out[f'{m}|{nm}']=dict(auc=auc,thr=float(thr),tpr=tpr,fpr=fpr,thrF1=float(thF))
json.dump(out,open('res_B5.json','w'),indent=1)
# logit percentiles
print('\nlogit percentiles (5,25,50,75,95):')
for m in models:
    d=D[m]
    for nm,z in [('speech clean',cat(d['speech_clean'])),('pause (digital -60 dBFS dither) in clean clips',cat(d['pause_clean'])),('noise-only -23 dBFS',d['nz'][-23]),('noise-only -40 dBFS',d['nz'][-40]),('noise-only -50 dBFS',d['nz'][-50]),('embedded stretch 20 dB',cat(d['stretch'][20])),('embedded stretch 5 dB',cat(d['stretch'][5])),('embedded stretch 0 dB',cat(d['stretch'][0]))]:
        print(f'{m:6s} {nm:48s}',np.percentile(z,[5,25,50,75,95]).round(2))
fig,axs=plt.subplots(1,2,figsize=(12,4),sharey=True)
for ax,m in zip(axs,models):
    d=D[m]; bins=np.linspace(-20,25,90)
    for nm,z,c in [('speech (clean clips)',cat(d['speech_clean']),'C0'),('pauses in clean clips',cat(d['pause_clean']),'C1'),('noise-only -23 dBFS',d['nz'][-23],'C3'),('noise-only -50 dBFS',d['nz'][-50],'C2'),('embedded stretch 5 dB',cat(d['stretch'][5]),'C4')]:
        ax.hist(np.clip(z,-20,25),bins=bins,density=True,histtype='step',label=nm,color=c,lw=1.6)
    ax.axvline(0,color='k',ls=':'); ax.set_title(m+': pre-sigmoid logit (0 = default threshold 0.5)'); ax.set_xlabel('logit'); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig('fig_logit_hist.png',dpi=110)
