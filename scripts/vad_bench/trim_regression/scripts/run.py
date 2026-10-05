"""usage: run.py VARIANTS(comma) KINDS(comma: talk,sinr,insert) [workers] [threads] [--conds a,b] [--only regex]"""
import os, re, sys, argparse, soundfile as sf
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__)); from common import *; from seg import *; import dec, variants
ap = argparse.ArgumentParser(); ap.add_argument("variants"); ap.add_argument("kinds")
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--conds", default=""); ap.add_argument("--only", default=""); ap.add_argument("--dets", default="ultra,redux,v3")
ap.add_argument("--noise-only", action="store_true", help="inserts: decode only segments overlapping the noise block")
a = ap.parse_args()
vs = a.variants.split(","); kinds = a.kinds.split(","); conds = a.conds.split(",") if a.conds else None
files = [m for m in manifest() if m["kind"] in kinds and m.get("dur", 99) > 30 and (conds is None or m["kind"] != "sinr" or m["cond"] in conds)
         and (not a.only or re.search(a.only, m["name"]))]
def need(det, m):
    total = sf.info(f"{W}/data/{m['name']}.wav").frames / 16000
    p = probs(det, m["name"]); out = set()
    for vn in vs:
        for s, e in segments(p, total, OPTS[det], variants.get(vn, det)):
            if a.noise_only and m["kind"] == "insert":
                lo, hi = m["ins"]
                if e <= lo - 0.5 or s >= hi + 0.5: continue
            out.add((round(s, 4), round(e, 4)))
    return sorted(out)
tasks = [(det, m) for det in a.dets.split(",") for m in files]
def go(t):
    det, m = t
    try: dec.run_segments(det, m["name"], need(det, m), threads=a.threads)
    except Exception as ex: print("FAIL", det, m["name"], ex, flush=True)
    return 1
print(len(tasks), "tasks", flush=True)
with ThreadPoolExecutor(a.workers) as ex:
    for i, _ in enumerate(ex.map(go, tasks)):
        if i % 50 == 0: print(i, flush=True)
print("done", flush=True)
