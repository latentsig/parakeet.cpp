import os
FUSION=os.environ.get('FUSION_DIR','../fusion')  # scripts/vad_bench/fusion with data/ built by prep_libri.py
import sys,numpy as np,json
from vadlib import *; from sig import *
os.makedirs('wav',exist_ok=True)
lib=list(np.load(FUSION+'/data/libri.npy',allow_pickle=True)); r=np.random.default_rng(0)
N=30*SR
sp=np.concatenate(lib[10:16]).astype(np.float32)[:N]
sp2=np.concatenate(lib[40:46]).astype(np.float32)[:N]
spec=ltass(lib[:60])
sigs={'speech_a':sp,'speech_b':sp2,'white':at_db(white(N,r),-25),'pink':at_db(colored(N,r,1),-25),'brown':at_db(colored(N,r,2),-25),
 'speechshaped':at_db(shaped(N,r,spec),-25),'silence':np.zeros(N,np.float32),'dither_1e-4':(r.standard_normal(N)*1e-4).astype(np.float32)}
rows=[]
for k,x in sigs.items():
    wr(f'wav/{k}.wav',x)
    for m,g in [('redux','redux'),('ultra','ultra'),('ultra','ultra-f16')]:
        p=ref(m).probs(sf.read(f'wav/{k}.wav',dtype='float32')[0]); c=cli_probs(g,f'wav/{k}.wav'); n=min(len(p),len(c)); e=np.abs(p[:n]-c[:n])
        rows.append(dict(sig=k,model=g,n=n,maxdiff=float(e.max()),meandiff=float(e.mean()),ref_speech_frac=float((p>=.5).mean()),cli_speech_frac=float((c>=.5).mean()),ref_mean_p=float(p.mean())))
        pass
json.dump(rows,open('res_A.json','w'),indent=1)

print('| signal | model file | frames | max abs diff | mean abs diff | speech frac ref | speech frac CLI |\n|---|---|---|---|---|---|---|')
for r in rows: print(f"| {r['sig']} | {r['model']} | {r['n']} | {r['maxdiff']:.2e} | {r['meandiff']:.2e} | {100*r['ref_speech_frac']:.1f}% | {100*r['cli_speech_frac']:.1f}% |")
