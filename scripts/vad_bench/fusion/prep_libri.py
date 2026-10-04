import io, numpy as np, soundfile as sf, os
os.environ["HF_HOME"]=os.path.abspath("hf")
from datasets import load_dataset, Audio
ds=load_dataset("openslr/librispeech_asr","clean",split="test",streaming=True).cast_column("audio",Audio(decode=False))
os.makedirs("data",exist_ok=True)
U=[];spk=[]
for i,ex in enumerate(ds):
    if i%13: continue
    y,sr=sf.read(io.BytesIO(ex["audio"]["bytes"]),dtype="float32"); assert sr==16000
    U.append(y);spk.append(int(ex["speaker_id"]))
np.save("data/libri.npy",np.array(U,dtype=object),allow_pickle=True); np.save("data/spk.npy",np.array(spk))
print(len(U),len(set(spk)),flush=True)
