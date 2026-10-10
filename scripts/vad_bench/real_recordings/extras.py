#!/usr/bin/env python3
"""Extras: (1) collar sensitivity of the frame metrics, (2) music false alarms by MUSAN source, (3) share of errors near reference boundaries."""
import sys, os, json, glob, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl, vr, systems
from rl import *
HEADS = ("ultra", "redux")
def ppfam(kind, h=None): return [s for s in SP if s[0] == "pp" and s[4][0] == kind and (h is None or s[4][1] == h)]
TP = {"sil": tune(ppfam("sil"))}
for h in HEADS:
    for k in ("head", "two", "gate", "org"): TP[(k, h)] = tune(ppfam(k, h))
SPECS = [("nat", "silero", 0.5, 0.25, 0.1, 0.03), ("head", "ultra", 0.5), ("head", "redux", 0.5)]
for h in HEADS: SPECS += [("two", h, 0.5, 0.5, 16, 16, 30), ("gate", h, 0.92)]
SPECS += [TP["sil"]] + [TP[(k, h)] for h in HEADS for k in ("head", "two", "gate", "org")]
COLL = (0.0, 0.1, 0.25)
D = {m["id"]: m for m in vr.load_recordings()}
def bnd_dist(ref):
    """distance in cells to the nearest reference transition"""
    n = len(ref); t = np.flatnonzero(np.diff(ref.astype(np.int8)) != 0) + 0.5
    if len(t) == 0: return np.full(n, 1e9)
    x = np.arange(n)[:, None] if False else np.arange(n)
    j = np.searchsorted(t, x); d = np.full(n, 1e9)
    l = np.where(j > 0, np.abs(x - t[np.maximum(j - 1, 0)]), 1e9); r = np.where(j < len(t), np.abs(t[np.minimum(j, len(t) - 1)] - x), 1e9)
    return np.minimum(l, r)
def work(m):
    if m["nonspeech"]: return m["id"], None
    r = systems.Rec(m); ref = m["ref"]; dist = bnd_dist(ref) * vr.GR; out = []
    for sp in SPECS:
        mk = systems.mask(r, sp); row = []
        for c in COLL:
            keep = dist >= c if c > 0 else np.ones(len(ref), bool)
            row.append(vr.counts(mk[keep], ref[keep]))
        # errors: fraction of fp / fn cells within 0.1 and 0.25 s of a boundary
        fp = mk & ~ref; fn = ~mk & ref
        row.append(np.array([fp.sum(), (fp & (dist < 0.1)).sum(), (fp & (dist < 0.25)).sum(), fn.sum(), (fn & (dist < 0.1)).sum(), (fn & (dist < 0.25)).sum()]))
        out.append(row)
    return m["id"], out
if __name__ == "__main__":
    ids = [m for m in D.values()]
    with Pool(6) as p: res = dict(p.map(work, ids, chunksize=1))
    lines = ["# Extras", "", "## 1. Collar sensitivity (cells within the collar of any reference speech/non-speech transition are ignored). Pooled F1, bootstrap CI over recordings (stratified by domain).", ""]
    lines.append("untuned systems on all recordings, tuned systems on the held-out split\n")
    lines.append("| system | split | F1 no collar | F1 collar 0.1 s | F1 collar 0.25 s | FP cells within 0.1 s of a boundary | FN cells within 0.1 s | FP within 0.25 s | FN within 0.25 s |\n|---|---|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(3)
    for si, sp in enumerate(SPECS):
        tuned = sp in [TP[k] for k in TP]
        split = "held" if tuned else None
        grp = [np.array([i for i, m in enumerate(M) if m["domain"] == d and not m["nonspeech"] and (split is None or m["split"] == split)]) for d in SPEECH_DOM]
        cells = []
        for ci in range(3):
            pt = sum(np.sum([res[M[i]["id"]][si][ci] for i in g], 0) for g in grp)
            bs = []
            for _ in range(500):
                tot = 0
                for g in grp:
                    b = rng.choice(g, size=len(g)); tot = tot + np.sum([res[M[i]["id"]][si][ci] for i in b], 0)
                bs.append(vr.prf(tot)[2])
            l, h = np.percentile(bs, [2.5, 97.5]); cells.append(f"{100*vr.prf(pt)[2]:.1f} [{100*l:.1f}, {100*h:.1f}]")
        e = sum(np.sum([res[M[i]["id"]][si][3] for i in g], 0) for g in grp)
        lines.append(f"| {sp} | {split or 'all'} | " + " | ".join(cells) + f" | {100*e[1]/e[0]:.0f}% | {100*e[4]/e[3]:.0f}% | {100*e[2]/e[0]:.0f}% | {100*e[5]/e[3]:.0f}% |")
    # music by source
    src = {}
    for j in glob.glob(f"{vr.ROOT}/data/musan/music_*.json"):
        src[os.path.basename(j)[:-5]] = json.load(open(j))["source"]
    lines += ["", "## 2. Music false alarms by MUSAN source (seconds called speech per hour of audio)", "", "| system | " + " | ".join(sorted(set(src.values()))) + " |", "|---|" + "---|" * len(set(src.values()))]
    for sp in [("nat", "silero", 0.5, 0.25, 0.1, 0.03), ("head", "ultra", 0.5), ("head", "redux", 0.5), ("two", "ultra", 0.5, 0.5, 16, 16, 30), ("two", "redux", 0.5, 0.5, 16, 16, 30), ("gate", "ultra", 0.92), ("gate", "redux", 0.92), ("gate", "redux", 0.98)]:
        row = []
        for sname in sorted(set(src.values())):
            ii = [i for i, m in enumerate(M) if m["domain"] == "music" and src[m["id"].replace("music_", "", 1)] == sname]
            if not ii: row.append("-"); continue
            k = IDX[sp]; fp = C[ii, k, 0] + C[ii, k, 1]; n = C[ii, k, :4].sum(1)
            row.append(f"{fp.sum()/n.sum()*3600:.0f} (n={len(ii)} files, {n.sum()*0.01/3600:.2f} h)")
        lines.append(f"| {sp} | " + " | ".join(row) + " |")
    open(f"{vr.ROOT}/results/extras.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
