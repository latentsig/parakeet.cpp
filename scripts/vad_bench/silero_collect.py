import json,os,subprocess,sys,numpy as np,soundfile as sf,onnxruntime as ort
# usage: collect.py CORE DATA_DIR MODEL_DIR PARAKEET_CLI WVAD
#   DATA_DIR: clips/ clips.json ted_talk.wav (make_clips.py, fetch_data.py)
#   MODEL_DIR: silero_vad.onnx silero-vad-f32.gguf silero-vad-f16.gguf ggml-silero-v6.2.3-ggml.bin ggml-silero-v6.2.0.bin
CORE,SB,M,PK,WV=sys.argv[1:6]
TS=["taskset","-c",CORE]
clips=json.load(open(SB+"/clips.json"))
files=[f"{SB}/clips/{c['name']}.wav" for c in clips]+[SB+"/ted_talk.wav"]
names=[c["name"] for c in clips]+["ted_talk"]
so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
sess=ort.InferenceSession(M+"/silero_vad.onnx",so,providers=["CPUExecutionProvider"])
def onnx_probs(y):
    n=(len(y)+511)//512; y=np.pad(y,(0,n*512-len(y)))
    st=np.zeros((2,1,128),np.float32); ctx=np.zeros(64,np.float32); out=[]
    for i in range(n):
        x=np.concatenate([ctx,y[i*512:(i+1)*512]])[None].astype(np.float32)
        o,st=sess.run(None,{"input":x,"state":st,"sr":np.array(16000,np.int64)})
        out.append(float(o[0,0])); ctx=x[0,-64:]
    return np.array(out)
res={}
def wh(model,tag):
    p=subprocess.run(TS+[WV,"probs",model,"1"]+files,capture_output=True,text=True,check=True).stdout.strip().split("\n")
    for nm,l in zip(names,p):
        j=json.loads(l); res.setdefault(nm,{})[tag]=np.array(j["probs"]); res[nm][tag+"_seg_default"]=j["default"]; res[nm][tag+"_seg_matched"]=j["matched"]
def pk(model,tag):
    for nm,f in zip(names,files):
        j=json.loads(subprocess.run(TS+[PK,"vad","--model",model,"--input",f,"--probabilities","--threads","1"],capture_output=True,text=True,check=True).stdout)
        res.setdefault(nm,{})[tag]=np.array(j["probabilities"]); res[nm][tag+"_seg_default"]=[[s["start"],s["end"]] for s in j["segments"]]
wh(M+"/ggml-silero-v6.2.3-ggml.bin","wcpp_conv623"); print("wcpp conv done",flush=True)
wh(M+"/ggml-silero-v6.2.0.bin","wcpp_repo620"); print("wcpp repo done",flush=True)
pk(M+"/silero-vad-f32.gguf","pk_f32"); print("pk f32",flush=True)
pk(M+"/silero-vad-f16.gguf","pk_f16"); print("pk f16",flush=True)
for nm,f in zip(names,files):
    y=sf.read(f,dtype="float32")[0]; res[nm]["onnx"]=onnx_probs(y)
print("onnx done",flush=True)
flat={}
for nm,d in res.items():
    for k,v in d.items():
        if isinstance(v,np.ndarray): flat[f"{nm}|{k}"]=v
np.savez_compressed("probs.npz",**flat)
json.dump({nm:{k:v for k,v in d.items() if not isinstance(v,np.ndarray)} for nm,d in res.items()},open("segs_own_default.json","w"))
