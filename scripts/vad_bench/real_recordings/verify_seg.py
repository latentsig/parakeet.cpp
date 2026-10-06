"""Compare the Python segmenter replica with `parakeet-cli vad --mode segments` (clean master build) on every recording and detector."""
import json, subprocess, sys, numpy as np
import vr
D = vr.load_recordings(); MOD = {"silero": "silero-vad-f16", "ultra": "ultra-vad-q8_0", "redux": "redux-vad"}
OPT = {"silero": (0.25, 0.1), "ultra": (0.1, 0.2), "redux": (0.1, 0.2)}
files = [m for i, m in enumerate(D) if i % int(sys.argv[1] if len(sys.argv) > 1 else 6) == 0]
tot = bad = 0
for m in files:
    P = vr.load_probs(m["id"])
    for k in MOD:
        for trim in (0.3, 0.0):
            r = subprocess.run([vr.CLI_CLEAN, "vad", "--model", f"{vr.MODELS}/{MOD[k]}.gguf", "--input", m["wav"], "--mode", "segments", "--trim", str(trim), "--threads", "4"], capture_output=True, text=True)
            j = json.loads(r.stdout); cli = [(s["start"], s["end"]) for s in j["segments"]]
            ms, mp = OPT[k]
            sp = vr.smooth_native(P[k], vr.FS[k], m["dur"], 0.5, 0.1, ms)
            mine, hard = vr.segment_by_mask(sp, vr.FS[k], m["dur"], min_pause=mp, trim=trim)
            ok = len(cli) == len(mine) and all(abs(a - c) < 1.1e-3 and abs(b - d) < 1.1e-3 for (a, b), (c, d) in zip(cli, mine))
            tot += 1; bad += not ok
            if not ok: print("MISMATCH", m["id"], k, trim, len(cli), len(mine))
print("checked", tot, "mismatch", bad)
