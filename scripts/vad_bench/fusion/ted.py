import sys,json,numpy as np,soundfile as sf,subprocess
import collect
from run import *
f=sys.argv[1]   # 16 kHz mono wav of one long talk
import onnxruntime as ort
so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
sess=ort.InferenceSession(collect.M,so,providers=["CPUExecutionProvider"])
y=sf.read(f,dtype="float32")[0]; n=int(round(len(y)/16000/GR))
d=dict(n=n,silero=hold(collect.sil(y,sess),0.032,n),snr_est=0.0)
for k,g in collect.HEADS.items():
    j=json.loads(subprocess.run([collect.CLI,"vad","--model",g,"--input",f,"--probabilities","--threads","1"],capture_output=True,text=True,check=True).stdout)
    d[k]=hold(np.array(j["probabilities"],np.float32),0.08,n)
tr=[i for i,x in enumerate(D) if not x["noise_only"]]
def agree(a,b):
    c=counts(a,b); tp,fp,fn,tn=c; P,R,F=prf(c); po=(tp+tn)/c.sum(); pe=((tp+fp)*(tp+fn)+(fn+tn)*(fp+tn))/c.sum()**2; return F,(po-pe)/(1-pe)
for h in("ultra","redux"):
    X=np.concatenate([feats(D[i],h,"lr6")[::10] for i in tr]); yy=np.concatenate([D[i]["ref_mask"][::10] for i in tr])
    mdl=LogisticRegression(C=1.0,max_iter=300).fit(X,yy)
    ms={"Silero .5":raw_mask("silero",(0.5,),d,h),f"{h} head .5":raw_mask("head",(0.5,),d,h),"OR":raw_mask("or",(0.5,0.5),d,h),"AND":raw_mask("and",(0.5,0.5),d,h),"mean .5":raw_mask("mean",(0.5,0.5),d,h),"two-stage .5/.5/160/160/300":raw_mask("two",(0.5,0.5,16,16,30),d,h),"LR+ctx @.5 (fit on synthetic)":mdl.predict_proba(feats(d,h,"lr6"))[:,1]>=0.5}
    ms={k:post(v,**POST_HEAD)[0] for k,v in ms.items()}
    print(f"## TED talk {len(y)/16000:.0f} s, head={h}: speech fraction and agreement (F1/kappa) with Silero and with the head")
    for k,v in ms.items():
        a=agree(v,ms["Silero .5"]); b=agree(v,ms[f"{h} head .5"]); print(f"| {k} | {100*v.mean():.1f}% | vs Silero {100*a[0]:.1f}/{a[1]:.3f} | vs head {100*b[0]:.1f}/{b[1]:.3f} |")
