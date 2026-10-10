#!/usr/bin/env python3
"""Run parakeet-cli vad --probabilities (clean master build) for every recording and detector; store probs (.npy) and the CLI's default speech regions (.json)."""
import glob, json, os, subprocess, sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
ROOT=os.environ.get("VAD_REAL_ROOT") or os.getcwd()
CLI_CLEAN=os.environ.get("PARAKEET_CLI", f"{ROOT}/parakeet-cli-clean"); MODELS=os.environ.get("VAD_REAL_MODELS", f"{ROOT}/models")
CLI=CLI_CLEAN; MOD={"silero":"silero-vad-f16","ultra":"ultra-vad-q8_0","redux":"redux-vad"}
os.makedirs(f"{ROOT}/probs",exist_ok=True)
jobs=[]
for pat in ("data/vox/*.json","data/ami/*.json","data/ava/*.json","data/musan/*.json","data/esc/*.json"):
    for j in sorted(glob.glob(f"{ROOT}/{pat}")):
        w=j[:-5]+".wav"
        if not os.path.exists(w): continue
        dom=pat.split("/")[1]; rid=os.path.basename(w)[:-4]
        if dom=="musan": dom=rid.split("_")[0]; rid=rid
        if dom in("vox","ami"): rid=f"{dom}_{rid}"
        elif dom=="ava": rid=f"ava_{rid}"
        else: rid=f"{dom}_{rid}"
        for k in MOD: jobs.append((w,rid,k))
def run(a):
    w,rid,k=a; out=f"{ROOT}/probs/{rid}.{k}.npy"
    if os.path.exists(out): return
    r=subprocess.run([CLI,"vad","--model",f"{MODELS}/{MOD[k]}.gguf","--input",w,"--probabilities","--threads","2"],capture_output=True,text=True)
    j=json.loads(r.stdout)
    np.save(out,np.array(j["probabilities"],np.float32)); json.dump({"segments":j["segments"],"duration":j["duration"],"frame_sec":j["frame_sec"]},open(f"{ROOT}/probs/{rid}.{k}.json","w"))
with ThreadPoolExecutor(8) as ex: list(ex.map(run,jobs))
print("done",len(jobs))
