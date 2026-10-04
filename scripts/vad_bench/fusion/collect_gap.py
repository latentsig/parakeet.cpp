import json,os,subprocess,numpy as np,soundfile as sf,onnxruntime as ort
from multiprocessing import Pool
# Paths come from the environment: SILERO_ONNX (silero_vad.onnx, 6.2.3), PARAKEET_CLI (parakeet-cli),
# ULTRA_GGUF and REDUX_GGUF (the ASR GGUFs of the two heads).
M=os.environ["SILERO_ONNX"]; CLI=os.environ["PARAKEET_CLI"]
HEADS={"ultra":os.environ["ULTRA_GGUF"],"redux":os.environ["REDUX_GGUF"]}
meta=json.load(open("gap.json"))
def sil(y,sess):
    n=(len(y)+511)//512; y=np.pad(y,(0,n*512-len(y)))
    st=np.zeros((2,1,128),np.float32); ctx=np.zeros(64,np.float32); out=[]
    for i in range(n):
        x=np.concatenate([ctx,y[i*512:(i+1)*512]])[None].astype(np.float32)
        o,st=sess.run(None,{"input":x,"state":st,"sr":np.array(16000,np.int64)})
        out.append(float(o[0,0])); ctx=x[0,-64:]
    return np.array(out,np.float32)
def work(m):
    so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
    sess=ort.InferenceSession(M,so,providers=["CPUExecutionProvider"])
    f=f"clips/{m['name']}.wav"; y=sf.read(f,dtype="float32")[0]; r={"silero":sil(y,sess)}
    for k,g in HEADS.items():
        j=json.loads(subprocess.run([CLI,"vad","--model",g,"--input",f,"--probabilities","--threads","1"],capture_output=True,text=True,check=True).stdout)
        r[k]=np.array(j["probabilities"],np.float32)
    return m["name"],r
if __name__=="__main__":
    with Pool(4) as p: res=p.map(work,meta,chunksize=4)
    flat={f"{n}|{k}":v for n,r in res for k,v in r.items()}
    np.savez_compressed("probs_gap.npz",**flat); print("done",len(res))
