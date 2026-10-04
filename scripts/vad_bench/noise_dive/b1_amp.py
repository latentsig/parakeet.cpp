import os
FUSION=os.environ.get('FUSION_DIR','../fusion')  # scripts/vad_bench/fusion with data/ built by prep_libri.py
import numpy as np,json
from vadlib import *; from sig import *
N=30*SR; r=np.random.default_rng(1)
lib=list(np.load(FUSION+'/data/libri.npy',allow_pickle=True)); spec=ltass(lib[:60])
base={'white':white(N,r),'pink':colored(N,r,1),'brown':colored(N,r,2),'speechshaped':shaped(N,r,spec)}
sp=np.concatenate(lib[10:16]).astype(np.float32)[:N]
base['speech']=sp
levels=list(range(-110,1,10))  # dBFS rms
res={}
print('| signal | '+' | '.join(f'{l}' for l in levels)+' |')
for k,b in base.items():
    row=[]
    for l in levels:
        x=at_db(b,l); x=np.clip(x,-1,1)   # float samples; clip like a wav would
        row.append({m:dict(sp=float((ref(m).probs(x)>=.5).mean()),mp=float(ref(m).probs(x).mean())) for m in ['redux','ultra']})
    res[k]=row
    for m in ['redux','ultra']:
        print(f'| {k} {m} | '+' | '.join(f'{100*c[m]["sp"]:.0f}' for c in row)+' |',flush=True)
# exact invariance test: pure gain on top of fixed noise (float arithmetic, no clipping), -80..0 dB, and x1000
b=at_db(base['white'],-20)
print('\nPure gain on white@-20dBFS (speech-frame %, max |logit change| vs 0 dB gain)')
ref0={m:ref(m).logits(b) for m in['redux','ultra']}
g={}
for gdb in [-80,-70,-60,-50,-40,-30,-20,-10,0]:
    x=(b*10**(gdb/20)).astype(np.float32)
    for m in ['redux','ultra']:
        z=ref(m).logits(x); g[(gdb,m)]=(float((z>=0).mean()),float(np.abs(z-ref0[m]).max()),float(np.abs(z-ref0[m]).mean()))
    print(gdb,{m:tuple(round(v,3) for v in g[(gdb,m)]) for m in['redux','ultra']})
x=(at_db(base['white'],-80)*1000).astype(np.float32)   # 1000x
b2=at_db(base['white'],-80)
for m in ['redux','ultra']:
    z=ref(m).logits(x); z0=ref(m).logits(b2); print('x1000 of -80dBFS noise',m,'speech%',100*(z>=0).mean(),'(base %',100*(z0>=0).mean(),') max|dz| to base',np.abs(z-z0).max())
json.dump(res,open('res_B1.json','w'))
