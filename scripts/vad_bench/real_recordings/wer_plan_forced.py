#!/usr/bin/env python3
"""Cost of a mid-speech cut: Silero speech mask cut at pauses with max segment 20 s (pause20) versus cut every 20 s regardless of pauses (forced20)."""
import json, numpy as np, soundfile as sf, vr
TALKS = ["GaryFlake-merged", "RobertGupta-merged", "EricMead_2009P_EricMead-merged", "DanBarber-merged", "MichaelSpecter-merged"]
plan = {"ultra": {}, "redux": {}}; stats = []
for t in TALKS:
    dur = sf.info(f"{vr.ROOT}/data/ted/{t}.wav").duration; P = np.load(f"{vr.ROOT}/probs/ted_{t.replace('-merged','')}.silero.npy")
    sp = vr.smooth_native(P, 0.032, dur, 0.5, 0.1, 0.25)
    a, ha = vr.segment_by_mask(sp, 0.032, dur, max_seg=20.0, min_pause=0.1)
    b, hb = vr.segment_by_mask(sp, 0.032, dur, max_seg=20.0, min_pause=1e6)
    stats.append((t, len(a), ha, len(b), hb))
    for h in plan: plan[h][t] = {"pause20": [[round(x, 4), round(y, 4)] for x, y in a], "forced20": [[round(x, 4), round(y, 4)] for x, y in b]}
print(stats); plan["_set"] = "ted"
json.dump(plan, open(f"{vr.ROOT}/results/wer_plan_forced.json", "w"))
