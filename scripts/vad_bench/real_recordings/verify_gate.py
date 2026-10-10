#!/usr/bin/env python3
"""Check the C++ run gate (SegmenterOpts::run_gate) against the study's Python gate on the same probabilities.

For every recording and detector in ROOT/probs it applies the gate in Python (vr.gate_frames: median of the
frames of each raw run of p >= 0.5, kept when >= gate), forms the speech regions with the repository's rules
(vr.native_mask), and compares them with the output of `gate_segtool` (built from gate_segtool.cpp and
src/vad_segmenter.cpp) on the same probabilities.

Build the tool from the repository root:
  g++ -O2 -std=c++17 -Isrc scripts/vad_bench/real_recordings/gate_segtool.cpp src/vad_segmenter.cpp -o gate_segtool
Run (VAD_REAL_ROOT holds probs/, as made by dump_probs.py; GATE_SEGTOOL is the binary; GATE_TMP a scratch dir):
  VAD_REAL_ROOT=work GATE_SEGTOOL=./gate_segtool python3 scripts/vad_bench/real_recordings/verify_gate.py
Prints the number of comparisons, the mismatches and the seconds of speech the gate removes per domain."""
import collections, glob, json, os, subprocess, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr

TOOL = os.environ["GATE_SEGTOOL"]
OPT = {"silero": (0.25, 0.1, 0.03), "ultra": (0.1, 0.2, 0.0), "redux": (0.1, 0.2, 0.0)}   # min speech, min pause, pad
GATES = {"ultra": [0.92, 0.96, 0.5], "redux": [0.92, 0.96, 0.98], "silero": [0.9, 0.97]}
tmp = tempfile.mkdtemp(prefix="gate_", dir=os.environ.get("GATE_TMP"))
raw = os.path.join(tmp, "p.f32")

def cpp(p, dur, kind, gate):
    p.astype(np.float32).tofile(raw)
    out = subprocess.run([TOOL, raw, repr(dur), "silero" if kind == "silero" else "head", repr(gate), "speech"],
                         capture_output=True, text=True, check=True).stdout.split()
    return np.array(out, float).reshape(-1, 2)

ids = sorted({os.path.basename(f).rsplit(".", 2)[0] for f in glob.glob(f"{vr.ROOT}/probs/*.npy")})
total = bad = 0
dropped = collections.defaultdict(lambda: [0.0, 0.0, 0])   # domain -> [C++ s, Python s, recordings] for Redux at 0.92
for rid in ids:
    for kind in ("ultra", "redux", "silero"):
        pj = [f"{vr.ROOT}/probs/{rid}.{k}.json" for k in (kind, "silero", "redux", "ultra")]
        pj = next((f for f in pj if os.path.exists(f)), None)
        if pj is None: continue
        dur = json.load(open(pj))["duration"]
        p = np.load(f"{vr.ROOT}/probs/{rid}.{kind}.npy").astype(np.float32)
        ms, mp, pad = OPT[kind]
        n10 = int(round(dur / vr.GR))
        for g in GATES[kind]:
            keep = vr.gate_frames(p, 0.5, g)
            _, st, en = vr.native_mask(np.where(keep, p, 0).astype(np.float32), vr.FS[kind], dur, 0.5, 0.1, ms, mp, pad, n10)
            c = cpp(p, dur, kind, g)
            ok = len(c) == len(st) and (len(c) == 0 or (np.abs(c[:, 0] - np.array(st)).max() < 1e-6 and np.abs(c[:, 1] - np.array(en)).max() < 1e-6))
            total += 1
            if not ok:
                bad += 1; print("MISMATCH", rid, kind, g, len(c), len(st))
        if kind == "redux":
            c0, c1 = cpp(p, dur, kind, 0.0), cpp(p, dur, kind, 0.92)
            _, s0, e0 = vr.native_mask(p, 0.08, dur, 0.5, 0.1, 0.1, 0.2, 0.0, n10)
            keep = vr.gate_frames(p, 0.5, 0.92)
            _, s1, e1 = vr.native_mask(np.where(keep, p, 0).astype(np.float32), 0.08, dur, 0.5, 0.1, 0.1, 0.2, 0.0, n10)
            a = dropped[rid.split("_")[0]]
            a[0] += (c0[:, 1] - c0[:, 0]).sum() - (c1[:, 1] - c1[:, 0]).sum()
            a[1] += (np.array(e0) - np.array(s0)).sum() - (np.array(e1) - np.array(s1)).sum()
            a[2] += 1
print("compared", total, "mismatches", bad)
for d, a in sorted(dropped.items()):
    print(f"{d}: {a[2]} recordings, Redux head at 0.92 removes {a[0]:.3f} s of speech (C++) and {a[1]:.3f} s (Python)")
