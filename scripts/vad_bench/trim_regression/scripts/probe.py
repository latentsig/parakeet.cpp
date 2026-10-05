import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from common import *
jobs = []
for det, (model, sil) in DET.items():
    os.makedirs(f"{W}/probs/{det}", exist_ok=True)
    for m in [x for x in manifest() if x.get("dur", 99) > 30]:
        out = f"{W}/probs/{det}/{m['name']}.f32"
        if os.path.exists(out): continue
        cmd = [CLI, "transcribe", "--model", model, "--input", f"{W}/data/{m['name']}.wav", "--json", "--threads", "2", "--vad"] + (["--vad-model", sil] if sil else [])
        jobs.append((out, cmd))
def run(j):
    out, cmd = j
    env = dict(os.environ, PK_SEGMENTS="/dev/null", PK_PROBOUT=out + ".tmp")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode == 0 and os.path.exists(out + ".tmp"): os.rename(out + ".tmp", out)
    else: print("FAIL", out, r.stderr[-200:], flush=True)
print(len(jobs), "jobs", flush=True)
with ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as ex: list(ex.map(run, jobs))
