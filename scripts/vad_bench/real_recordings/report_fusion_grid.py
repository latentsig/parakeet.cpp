#!/usr/bin/env python3
"""Fusion extension sizes: pooled F1 (all recordings, ts .5 th .5, default post) and non-speech FA, as a function of pre/post/fill. Shows which part of the fusion carries the gain."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl
from rl import *
out = ["# Fusion rule, ts = th = .5, default post, all recordings: delta F1 (pp) versus Silero .5 with the same post, paired bootstrap over recordings; FA = seconds called speech per hour on music/noise/esc pooled", "",
       "| head | pre cells (10 ms) | post cells | fill cells | pooled F1 | dF1 vs Silero .5 [95% CI] | non-speech FA s/h |", "|---|---|---|---|---|---|---|"]
g = [sel((d,)) for d in SPEECH_DOM]
def fa(s):
    ii = sel(NS_DOM); k = IDX[s]; return (C[ii, k, 1].sum()) / (C[ii, k, :4].sum()) * 3600
for h in ("ultra", "redux"):
    for pre, po, Y in ((0, 0, 0), (16, 0, 0), (32, 0, 0), (0, 16, 0), (0, 32, 0), (0, 0, 30), (0, 0, 60), (16, 16, 30), (32, 16, 30), (32, 32, 60), (16, 16, 60)):
        s = ("two", h, 0.5, 0.5, pre, po, Y); pt, bs = stat(s, g); d, db = delta(s, ("sil", 0.5), g)
        out.append(f"| {h} | {pre} | {po} | {Y} | {fmt(pt, bs)} | {fmt(d, db, 100, 2)} | {fa(s):.0f} |")
open(f"{rl.vr.ROOT}/results/fusion_grid.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
