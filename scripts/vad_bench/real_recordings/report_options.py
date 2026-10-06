#!/usr/bin/env python3
"""One-at-a-time option sweeps (no selection involved): Silero native options and head thresholds. All recordings. F1 per domain + pooled with CIs, FA on non-speech."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl
from rl import *
out = ["# Option sweeps, all recordings, one option changed from the Silero defaults (thr .5, min speech .25 s, min pause/silence .1 s, pad .03 s)", "",
       "| option | value | vox F1 | ami F1 | ava F1 | pooled F1 | pooled P | pooled R | non-speech FA s/h (music / noise / esc) |", "|---|---|---|---|---|---|---|---|---|"]
def fa(spec, d):
    ii = sel((d,)); k = IDX[spec]; fp = C[ii, k, 1]; n = C[ii, k, :4].sum(1); return fp.sum() / n.sum() * 3600
def row(opt, val, spec):
    cells = [fmt(*stat(spec, [sel((d,))])) for d in SPEECH_DOM]; g = [sel((d,)) for d in SPEECH_DOM]
    out.append(f"| {opt} | {val} | " + " | ".join(cells) + f" | {fmt(*stat(spec, g))} | {fmt(*stat(spec, g, 0))} | {fmt(*stat(spec, g, 1))} | " + " / ".join(f"{fa(spec, d):.0f}" for d in NS_DOM) + " |")
D0 = (0.5, 0.25, 0.1, 0.03)
sp = lambda t=D0[0], ms=D0[1], mp=D0[2], pad=D0[3]: ("nat", "silero", t, ms, mp, pad)
row("(defaults)", "-", sp())
for t in (0.1, 0.2, 0.3, 0.4, 0.6, 0.7): row("threshold", t, sp(t=t))
for ms in (0.1, 0.5): row("min speech s", ms, sp(ms=ms))
for mp in (0.2, 0.5): row("min silence s (pause)", mp, sp(mp=mp))
for pad in (0.0, 0.1, 0.2): row("speech pad s", pad, sp(pad=pad))
out += ["", "## Silero threshold with the unified post (ms .1, pause .2, pad 0), thresholds below .1", "", "| thr | vox F1 | ami F1 | ava F1 | pooled F1 | nonspeech FA s/h |", "|---|---|---|---|---|---|"]
for t in (0.02, 0.05, 0.075, 0.1, 0.2):
    s = ("sil", t); out.append(f"| {t} | " + " | ".join(fmt(*stat(s, [sel((d,))])) for d in SPEECH_DOM) + f" | {fmt(*stat(s, [sel((d,)) for d in SPEECH_DOM]))} | " + " / ".join(f"{fa(s, d):.0f}" for d in NS_DOM) + " |")
out += ["", "## Head thresholds (unified post: bridge .1, min speech .1, pause .2, pad 0), all recordings", "", "| head | thr | vox F1 | ami F1 | ava F1 | pooled F1 | pooled P | pooled R | non-speech FA s/h (music / noise / esc) |", "|---|---|---|---|---|---|---|---|---|"]
for h in ("ultra", "redux"):
    for t in (0.3, 0.5, 0.7, 0.9, 0.95):
        s = ("head", h, float(t)); g = [sel((d,)) for d in SPEECH_DOM]
        out.append(f"| {h} | {t} | " + " | ".join(fmt(*stat(s, [sel((d,))])) for d in SPEECH_DOM) + f" | {fmt(*stat(s, g))} | {fmt(*stat(s, g, 0))} | {fmt(*stat(s, g, 1))} | " + " / ".join(f"{fa(s, d):.0f}" for d in NS_DOM) + " |")
open(f"{rl.vr.ROOT}/results/options.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
