import os
FUSION=os.environ.get('FUSION_DIR','../fusion')  # scripts/vad_bench/fusion with data/ built by prep_libri.py
import numpy as np,soundfile as sf,sys,onnxruntime as ort,os
sys.path.insert(0,FUSION); import fl
from vadlib import *
x=sf.read('talk/JaneMcGonigal-merged.wav',dtype='float32')[0]
so=ort.SessionOptions(); so.intra_op_num_threads=2
sess=ort.InferenceSession(os.environ['SILERO_ONNX'],so,providers=["CPUExecutionProvider"])
y=np.pad(x,(0,(-len(x))%512)); st=np.zeros((2,1,128),np.float32); ctx=np.zeros(64,np.float32); sil=[]
for i in range(len(y)//512):
    xx=np.concatenate([ctx,y[i*512:(i+1)*512]])[None].astype(np.float32); o,st=sess.run(None,{"input":xx,"state":st,"sr":np.array(16000,np.int64)}); sil.append(float(o[0,0])); ctx=xx[0,-64:]
sil=np.array(sil); n=int(len(x)/160)
S,_=fl.post(fl.hold(sil,0.032,n)>=0.5,**fl.POST_SIL)
print('talk %.0f s; Silero speech time %.1f%%'%(len(x)/16000,100*S.mean()))
for m in ['redux','ultra']:
    z=logits_blocks(m,x); base,_=fl.post(fl.hold(sig(z),0.08,n)>=0.5,**fl.POST_HEAD)
    def rc(q):
        keep=np.zeros(len(z),bool)
        for i,j in zip(*fl.runs(z>=0)):
            if np.median(z[i:j])>=q: keep[i:j]=True
        return fl.post(fl.hold(keep.astype(np.float32),0.08,n)>=0.5,**fl.POST_HEAD)[0]
    print(f'\n{m}: speech time % | agreement with Silero (F1 of mask vs Silero mask) | frames removed vs head 0.5 (% of its speech frames)')
    for nm,mk in [('head 0.5',base),('head 0.7',fl.post(fl.hold(sig(z),0.08,n)>=0.7,**fl.POST_HEAD)[0]),('head 0.9',fl.post(fl.hold(sig(z),0.08,n)>=0.9,**fl.POST_HEAD)[0]),('runconf 2.5',rc(2.5)),('runconf 3.5',rc(3.5))]:
        P,R,F=fl.prf(fl.counts(mk,S)); print(f'  {nm:12s} {100*mk.mean():5.1f}% | F1 vs Silero {100*F:.1f} (P {100*P:.1f} R {100*R:.1f}) | removed {100*((base&~mk).sum()/base.sum()):.2f}%')
