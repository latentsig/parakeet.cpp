#!/usr/bin/env python3
"""Build segment lists for the WER experiment from probability dumps (replica of segment_by_vad, trim .3), check the three native ones against the CLI, write results/wer_plan.json."""
import json, subprocess, sys, os, numpy as np
import vr, systems
SET = sys.argv[1] if len(sys.argv) > 1 else "ted"
TALKS = ["GaryFlake-merged", "RobertGupta-merged", "EricMead_2009P_EricMead-merged", "DanBarber-merged", "MichaelSpecter-merged"] if SET == "ted" else ["GaryFlake+ins", "RobertGupta+ins", "EricMead_2009P_EricMead+ins"]
MOD = {"silero": "silero-vad-f16", "ultra": "ultra-vad-q8_0", "redux": "redux-vad"}
class R:   # minimal recording
    pass
def rec(t):
    import soundfile as sf
    dur = sf.info(f"{vr.ROOT}/data/{SET}/{t}.wav").duration
    meta = dict(id=("ted_" + t.replace("-merged", "")) if SET == "ted" else ("comp_" + t), ref=np.zeros(int(round(dur / vr.GR)), bool), dur=dur)
    return systems.Rec(meta), dur
plan = {"ultra": {}, "redux": {}}; bad = 0; chk = 0
if SET != "ted":
    import subprocess as sp_
    for t in TALKS:
        for k in MOD:
            if os.path.exists(f"{vr.ROOT}/probs/comp_{t}.{k}.npy"): continue
            j = json.loads(sp_.run([vr.CLI_CLEAN, "vad", "--model", f"{vr.MODELS}/{MOD[k]}.gguf", "--input", f"{vr.ROOT}/data/{SET}/{t}.wav", "--probabilities", "--threads", "4"], capture_output=True, text=True).stdout)
            np.save(f"{vr.ROOT}/probs/comp_{t}.{k}.npy", np.array(j["probabilities"], np.float32))
for t in TALKS:
    r, dur = rec(t)
    S = vr.smooth_native(r.P["silero"], vr.FS["silero"], dur, 0.5, 0.1, 0.25)
    seg = {"silero": vr.segment_by_mask(S, vr.FS["silero"], dur, min_pause=0.1)[0]}
    for h in ("ultra", "redux"):
        H = vr.smooth_native(r.P[h], vr.FS[h], dur, 0.5, 0.1, 0.1)
        seg[h] = vr.segment_by_mask(H, vr.FS[h], dur, min_pause=0.2)[0]
        for name, spec in (("fusion_def", ("two", h, 0.5, 0.5, 16, 16, 30)), ("fusion_tuned", ("two", h, 0.1, 0.3, 32, 8, 30)), ("gate92", ("gate", h, 0.92)), ("orgate92", ("org", h, 0.92))):
            seg[f"{name}_{h}"] = vr.segment_by_mask(systems.mask(r, spec), vr.GR, dur, min_pause=0.2)[0]
    # CLI check of the three native segmentations
    for k in MOD:
        out = json.loads(subprocess.run([vr.CLI_CLEAN, "vad", "--model", f"{vr.MODELS}/{MOD[k]}.gguf", "--input", f"{vr.ROOT}/data/{SET}/{t}.wav", "--mode", "segments", "--threads", "4"], capture_output=True, text=True).stdout)
        cli = [(s["start"], s["end"]) for s in out["segments"]]; mine = seg[k]; chk += 1
        ok = len(cli) == len(mine) and all(abs(a - c) < 1.1e-3 and abs(b - d) < 1.1e-3 for (a, b), (c, d) in zip(cli, mine))
        bad += not ok; print(t, k, "cli segs", len(cli), "replica", len(mine), "match", ok, flush=True)
    for h in ("ultra", "redux"):
        plan[h][t] = {"head": seg[h], "silero": seg["silero"], **{n: seg[f"{n}_{h}"] for n in ("fusion_def", "fusion_tuned", "gate92", "orgate92")}}
        plan[h][t] = {k: [[round(a, 4), round(b, 4)] for a, b in v] for k, v in plan[h][t].items()}
print("CLI vs replica: checked", chk, "mismatch", bad)
plan["_set"] = SET
json.dump(plan, open(f"{vr.ROOT}/results/wer_plan.json" if SET == "ted" else f"{vr.ROOT}/results/wer_plan_{SET}.json", "w"))
