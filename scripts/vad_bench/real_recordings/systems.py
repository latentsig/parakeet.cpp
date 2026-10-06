"""System definitions: spec tuple -> final 10 ms speech mask for one recording.

A recording object `r` has: n (10 ms cells), dur, P = {silero,ultra,redux} native-frame probabilities, G = hold()-grid of each.
All 'unified' systems use the same post (bridge .1, min speech .1, min pause .2, pad 0); 'nat' systems use each detector's own options
through the exact native-frame replica of speech_regions()."""
import numpy as np
import vr
from vr import FS, GR

class Rec:
    def __init__(self, meta):
        self.m = meta; self.id = meta["id"]; self.n = len(meta["ref"]); self.dur = meta["dur"]
        self.P = vr.load_probs(self.id)
        self.G = {k: vr.hold(self.P[k], FS[k], self.n) for k in FS}
        self.gate_cache = {}
    def gated(self, h, med):
        """gated head probability on the grid: p where the raw p>=0.5 run has median >= med, else 0"""
        key = (h, med)
        if key not in self.gate_cache:
            keep = vr.gate_frames(self.P[h], 0.5, med)
            g = np.where(keep, self.P[h], 0).astype(np.float32)
            self.gate_cache[key] = vr.hold(g, FS[h], self.n)
        return self.gate_cache[key]

def mask(r, spec, pp=None):
    pp = vr.POST_HEAD if pp is None else pp
    k = spec[0]
    if k == "pp":      # ("pp", min_speech, min_pause, pad, inner_spec): inner system with its own post-processing
        return mask(r, spec[4], dict(bridge=0.1, min_speech=spec[1], min_pause=spec[2], pad=spec[3]))
    if k == "sil":      # unified post: ("sil", thr)
        return vr.post(r.G["silero"] >= spec[1], **pp)
    if k == "head":     # ("head", model, thr)
        return vr.post(r.G[spec[1]] >= spec[2], **pp)
    if k == "nat":      # native replica: ("nat", model, thr, min_speech, min_pause, pad)
        _, h, thr, ms, mp, pad = spec
        return vr.native_mask(r.P[h], FS[h], r.dur, thr, 0.1, ms, mp, pad, r.n)[0]
    if k == "or":       # ("or", model, ts, th)
        return vr.post((r.G["silero"] >= spec[2]) | (r.G[spec[1]] >= spec[3]), **pp)
    if k == "and":
        return vr.post((r.G["silero"] >= spec[2]) & (r.G[spec[1]] >= spec[3]), **pp)
    if k == "mean":     # ("mean", model, w, thr)
        return vr.post((spec[2] * r.G["silero"] + (1 - spec[2]) * r.G[spec[1]]) >= spec[3], **pp)
    if k == "two":      # ("two", model, ts, th, pre, post, Y)  cells of 10 ms
        _, h, ts, th, pre, po, Y = spec
        return vr.post(vr.two_stage(r.G["silero"] >= ts, r.G[h] >= th, pre, po, Y), **pp)
    if k == "gate":     # ("gate", model, med)  head thr .5 with run median gate
        return vr.post(r.gated(spec[1], spec[2]) >= 0.5, **pp)
    if k == "twog":     # two-stage whose head input is the gated head: ("twog", model, med, pre, post, Y)
        _, h, med, pre, po, Y = spec
        return vr.post(vr.two_stage(r.G["silero"] >= 0.5, r.gated(h, med) >= 0.5, pre, po, Y), **pp)
    if k == "org":      # OR(Silero .5, gated head): ("org", model, med)
        return vr.post((r.G["silero"] >= 0.5) | (r.gated(spec[1], spec[2]) >= 0.5), **pp)
    if k == "ref":      # oracle: the reference mask itself (post applied to keep the same smoothing)
        return vr.post(r.m["ref"], **pp)
    raise ValueError(spec)
