"""Decode segments through the (patched) CLI with a per-(det,file) cache of segment -> words."""
import json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__)); from common import *
def key(s, e): return f"{s:.4f},{e:.4f}"
def cpath(det, name): return f"{W}/cache/{det}/{name}.json"
def load_cache(det, name):
    f = cpath(det, name)
    return json.load(open(f)) if os.path.exists(f) else {}
def run_segments(det, name, segs, threads=5, tmpdir=None):
    cache = load_cache(det, name)
    miss = sorted({(round(s, 4), round(e, 4)) for s, e in segs if key(s, e) not in cache})
    if not miss: return cache
    model, sil = DET[det]
    with tempfile.TemporaryDirectory(dir=f"{W}/tmp") as td:
        sf, so = f"{td}/seg.txt", f"{td}/out.txt"
        open(sf, "w").write("".join(f"{s:.4f} {e:.4f}\n" for s, e in miss))
        cmd = [CLI, "transcribe", "--model", model, "--input", f"{W}/data/{name}.wav", "--json", "--threads", str(threads), "--vad"] + (["--vad-model", sil] if sil else [])
        r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PK_SEGMENTS=sf, PK_SEGOUT=so))
        if r.returncode != 0: raise RuntimeError(r.stderr[-300:])
        lines = open(so).read().split("\n") if os.path.exists(so) else []
    lines = [l for l in lines if l]
    # one line per slice, in the order of the segment file; slices of no speech never happen here (all kept)
    assert len(lines) == len(miss), (len(lines), len(miss), name)
    for (s, e), l in zip(miss, lines):
        a, b, w = (l.split("\t") + [""])[:3]
        words = []
        import re as _re
        for tok in _re.split(r" (?=-?\d+\.\d{3}\|-?\d+\.\d{3}\|\d\.\d{3}\|)", w):
            if tok:
                st, en, cf, tx = tok.split("|", 3); words.append([float(st), float(en), float(cf), tx])
        cache[key(s, e)] = dict(out=[float(a), float(b)], words=words)
    os.makedirs(os.path.dirname(cpath(det, name)), exist_ok=True)
    json.dump(cache, open(cpath(det, name) + ".tmp", "w")); os.rename(cpath(det, name) + ".tmp", cpath(det, name))
    return cache
