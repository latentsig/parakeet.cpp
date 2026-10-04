import numpy as np,json,torch
from vadlib import *; from sig import *; import corpus
S=np.load('fixed_stats.npz'); T=lambda a: torch.tensor(a)
variants={'file':(None,None),'fixedA':(T(S['aM']),T(S['aS'])),'fixedB':(T(S['bM']),T(S['bS']))}
models=['redux','ultra']
cl=corpus.clean_clips(); gaps=corpus.gap_clips()
R={}
def f1(p,r):
    n=min(len(p),len(r)); p=p[:n]; r=r[:n]; tp=(p&r).sum(); fp=(p&~r).sum(); fn=(~p&r).sum()
    P=tp/max(1,tp+fp); Rc=tp/max(1,tp+fn); return P,Rc,2*P*Rc/max(1e-9,P+Rc)
# 1. clean speech clips
for v,(mu,sd) in variants.items():
    for m in models:
        c=np.zeros(3)
        for y,spans in cl:
            z=logits_blocks(m,y,mu,sd) if mu is None else ref(m).logits(y,mu,sd)
            p=z>=0; r=frame_mask(spans,len(p)); n=len(p)
            c+=[(p&r).sum(),(p&~r).sum(),(~p&r).sum()]
        P=c[0]/(c[0]+c[1]); Rc=c[0]/(c[0]+c[2]); R[f'clean|{v}|{m}']=dict(P=P,R=Rc,F1=2*P*Rc/(P+Rc))
        print('clean',v,m,R[f'clean|{v}|{m}'],flush=True)
# 2. noise-only 30 s at absolute levels
rng=np.random.default_rng(5); N=30*SR
for kind in ['white','pink']:
    for lvl in [-60,-50,-40,-30,-23,-15]:
        fr={v:{m:[] for m in models} for v in variants}
        for k in range(6):
            b=at_db(white(N,rng) if kind=='white' else colored(N,rng,1),lvl)
            for v,(mu,sd) in variants.items():
                for m in models: fr[v][m].append((ref(m).logits(b,mu,sd)>=0).mean())
        for v in variants:
            for m in models: R[f'nz|{kind}|{lvl}|{v}|{m}']=float(np.mean(fr[v][m]))
        print('noise-only',kind,lvl,{v:{m:round(100*np.mean(fr[v][m]),1) for m in models} for v in variants},flush=True)
# 3. embedded stretch
agg={}
for g in gaps:
    y=g['y']; ref_m=None
    for v,(mu,sd) in variants.items():
        for m in models:
            z=logits_blocks(m,y,mu,sd) if mu is None else ref(m).logits(y,mu,sd); p=z>=0
            n=len(p); t=(np.arange(n)+.5)*.08; st=(t>=g['stretch'][0])&(t<g['stretch'][1]); sp=frame_mask(g['spans'],n)
            a=agg.setdefault((g['kind'],g['snr'],v,m),dict(fp=[],tp=0,fn=0,fpall=0))
            a['fp'].append(p[st].mean()); a['tp']+=(p&sp).sum(); a['fn']+=(~p&sp).sum(); a['fpall']+=(p&~sp&~st).sum()
rows=[]
for (kind,snr,v,m),a in sorted(agg.items()):
    R[f'gap|{kind}|{snr}|{v}|{m}']=dict(fp_stretch=float(np.mean(a['fp'])),recall=a['tp']/(a['tp']+a['fn']))
tab={}
for m in models:
    print(f'\n{m}: embedded 30 s stretch false-alarm frame rate % (speech recall % in brackets), white+pink pooled')
    print('| SNR | '+' | '.join(variants)+' |'); print('|---|'+'---|'*len(variants))
    for snr in [20,10,5,0]:
        cells=[]
        for v in variants:
            fp=np.mean([R[f'gap|{k}|{snr}|{v}|{m}']['fp_stretch'] for k in ['white','pink']]); rc=np.mean([R[f'gap|{k}|{snr}|{v}|{m}']['recall'] for k in ['white','pink']])
            cells.append(f'{100*fp:.1f} ({100*rc:.0f})')
        print(f'| {snr} dB | '+' | '.join(cells)+' |')
json.dump(R,open('res_B3.json','w'),indent=1)
