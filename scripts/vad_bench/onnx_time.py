import sys,time,numpy as np,soundfile as sf,onnxruntime as ort
so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
s=ort.InferenceSession(sys.argv[1],so,providers=["CPUExecutionProvider"])
y=sf.read(sys.argv[2],dtype="float32")[0]; n=(len(y)+511)//512; y=np.pad(y,(0,n*512-len(y)))
st=np.zeros((2,1,128),np.float32); ctx=np.zeros(64,np.float32); sr=np.array(16000,np.int64)
t0=time.perf_counter()
for i in range(n):
    x=np.concatenate([ctx,y[i*512:(i+1)*512]])[None]
    o,st=s.run(None,{"input":x,"state":st,"sr":sr}); ctx=x[0,-64:]
print("%.4f"%(time.perf_counter()-t0))
