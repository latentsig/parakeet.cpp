import os
FUSION=os.environ.get('FUSION_DIR','../fusion')  # scripts/vad_bench/fusion with data/ built by prep_libri.py
import numpy as np,json
from vadlib import *; from sig import *
N=30*SR; r=np.random.default_rng(3)
lib=list(np.load(FUSION+'/data/libri.npy',allow_pickle=True))
S={}
S['digital silence']=np.zeros(N,np.float32)
S['DC offset 0.1']=np.full(N,0.1,np.float32)
for f in [100,440,1000,3000]: S[f'sine {f} Hz -20 dBFS']=sine(N,f)
S['sine 1 kHz -60 dBFS']=sine(N,1000,-60)
S['sine sweep 50-7500 Hz (30 s)']=sweep(N)
S['sine sweep -50 dBFS']=sweep(N,db=-50)
S['clicks 3/s -12 dBFS']=clicks(N,r,3,-12)
S['clicks 10/s -12 dBFS']=clicks(N,r,10,-12)
S['clicks 1/s -12 dBFS']=clicks(N,r,1,-12)
S['hum 50 Hz+harmonics -25 dBFS']=hum(N,50,-25)
S['hum 60 Hz+harmonics -25 dBFS']=hum(N,60,-25)
S['hum 50 Hz + white -60 dBFS']=hum(N,50,-25)+at_db(white(N,r),-60)
S['music-like chords -20 dBFS']=music(N,r)
S['white -25 dBFS']=at_db(white(N,r),-25); S['pink -25 dBFS']=at_db(colored(N,r,1),-25); S['brown -25 dBFS']=at_db(colored(N,r,2),-25)
S['white -70 dBFS']=at_db(white(N,r),-70)
S['speech (LibriSpeech x6)']=np.concatenate(lib[10:16]).astype(np.float32)[:N]
rows=[]; Z={}
print('| signal | Redux speech frames % | Redux mean p | Ultra speech frames % | Ultra mean p |\n|---|---|---|---|---|')
for k,x in S.items():
    cell=[]
    for m in ['redux','ultra']:
        z=ref(m).logits(x.astype(np.float32)); Z[f'{k}|{m}']=z; p=sig(z); cell+= [f'{100*(p>=.5).mean():.1f}',f'{p.mean():.2f}']
    print(f'| {k} | '+' | '.join(cell)+' |',flush=True)
np.savez('res_B2_logits.npz',**{k.replace('|','@'):v for k,v in Z.items()})
