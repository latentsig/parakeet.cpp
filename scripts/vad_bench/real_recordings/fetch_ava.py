#!/usr/bin/env python3
"""AVA-Speech (human labels, film audio) from nccratliri/vad-human-ava-speech: download N clips by hash selection, convert to 16k mono int16."""
import json, os, sys, hashlib, urllib.request, subprocess
from huggingface_hub import HfApi
outd=sys.argv[1]; N=int(sys.argv[2])
os.environ.setdefault("HF_HOME",os.path.join(os.environ.get("VAD_REAL_ROOT") or os.getcwd(),"hfhome"))
api=HfApi(); files=[f for f in api.list_repo_files("nccratliri/vad-human-ava-speech",repo_type="dataset") if f.endswith(".wav")]
files.sort(key=lambda f: hashlib.md5(f.encode()).hexdigest())
for f in files[:N]:
    rid=os.path.basename(f)[:-4]
    for ext in ("wav","json"):
        u=f"https://huggingface.co/datasets/nccratliri/vad-human-ava-speech/resolve/main/{f[:-3]}{ext}"
        tmp=f"{outd}/{rid}.raw.{ext}"
        urllib.request.urlretrieve(u,tmp)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",f"{outd}/{rid}.raw.wav","-ac","1","-ar","16000","-c:a","pcm_s16le",f"{outd}/{rid}.wav"],check=True)
    os.remove(f"{outd}/{rid}.raw.wav"); os.rename(f"{outd}/{rid}.raw.json",f"{outd}/{rid}.json")
    print(rid,flush=True)
