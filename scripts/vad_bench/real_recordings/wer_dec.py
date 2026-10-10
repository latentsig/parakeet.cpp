#!/usr/bin/env python3
"""WER part: decode explicit segment lists through the throwaway-patched CLI (PK_SEGMENTS), cache decoded words per (asr model, talk, segment).
usage: wer_dec.py <plan.json>   plan = {asr: {talk: {system: [[s,e],...]}}}  (made by wer_plan.py)"""
import json, os, re, subprocess, sys, tempfile
ROOT = os.environ.get("VAD_REAL_ROOT") or os.getcwd()
MODELS = os.environ.get("VAD_REAL_MODELS", f"{ROOT}/models")
CLI = os.environ.get("PARAKEET_CLI_PATCHED", f"{ROOT}/parakeet-cli-patched")
MODEL = {"ultra": f"{MODELS}/ultra-q8_0.gguf", "redux": f"{MODELS}/redux-packed.gguf"}
def key(s, e): return f"{s:.4f},{e:.4f}"
SET = "ted"
def cpath(asr, talk): return f"{ROOT}/results/wercache/{asr}/{talk}.json" if SET == "ted" else f"{ROOT}/results/wercache_{SET}/{asr}/{talk}.json"
def load(asr, talk):
    f = cpath(asr, talk); return json.load(open(f)) if os.path.exists(f) else {}
def run(asr, talk, segs, threads):
    cache = load(asr, talk)
    miss = sorted({(round(s, 4), round(e, 4)) for s, e in segs if key(s, e) not in cache})
    if not miss: return
    with tempfile.TemporaryDirectory(dir=f"{ROOT}/results") as td:
        sf, so = f"{td}/seg.txt", f"{td}/out.txt"
        open(sf, "w").write("".join(f"{s:.4f} {e:.4f}\n" for s, e in miss))
        r = subprocess.run([CLI, "transcribe", "--model", MODEL[asr], "--input", f"{ROOT}/data/{SET}/{talk}.wav", "--json", "--threads", str(threads), "--vad"],
                           capture_output=True, text=True, env=dict(os.environ, PK_SEGMENTS=sf, PK_SEGOUT=so))
        if r.returncode != 0: raise RuntimeError(r.stderr[-300:])
        lines = [l for l in open(so).read().split("\n") if l] if os.path.exists(so) else []
    assert len(lines) == len(miss), (len(lines), len(miss), talk)
    for (s, e), l in zip(miss, lines):
        a, b, w = (l.split("\t") + [""])[:3]; words = []
        for tok in re.split(r" (?=-?\d+\.\d{3}\|-?\d+\.\d{3}\|\d\.\d{3}\|)", w):
            if tok:
                st, en, cf, tx = tok.split("|", 3); words.append([float(st), float(en), float(cf), tx])
        cache[key(s, e)] = dict(out=[float(a), float(b)], words=words)
    os.makedirs(os.path.dirname(cpath(asr, talk)), exist_ok=True)
    json.dump(cache, open(cpath(asr, talk) + ".tmp", "w")); os.rename(cpath(asr, talk) + ".tmp", cpath(asr, talk))
if __name__ == "__main__":
    plan = json.load(open(sys.argv[1])); threads = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    SET = plan.pop("_set", "ted")
    for asr, talks in plan.items():
        for talk, systems in talks.items():
            allsegs = [tuple(x) for segs in systems.values() for x in segs]
            run(asr, talk, allsegs, threads); print("decoded", asr, talk, len(set(allsegs)), flush=True)
