#!/usr/bin/env python3
"""Composite long recordings: a real TED talk with 3 real non-speech clips (MUSAN music x2, noise x1; 40/30/40 s) inserted at Silero pauses near 25/50/75 percent.
The reference transcript is the talk's own transcript (inserted clips contain no scored speech), so every inserted word counts as an insertion."""
import glob, json, os, numpy as np, soundfile as sf
import vr
ROOT = vr.ROOT
src = {os.path.basename(j)[:-5]: json.load(open(j))["source"] for j in glob.glob(f"{ROOT}/data/musan/music_*.json")}
pick = {"fma": sorted(k for k, v in src.items() if v == "fma")[0], "jamendo": sorted(k for k, v in src.items() if v == "jamendo")[1], "noise": "noise_01"}
print(pick)
def clip(name, sec, off=20):
    y, sr = sf.read(f"{ROOT}/data/musan/{name}.wav", dtype="float32"); assert sr == 16000
    y = y[off * 16000: off * 16000 + sec * 16000]; return y
os.makedirs(f"{ROOT}/data/comp", exist_ok=True)
for t in ["GaryFlake-merged", "RobertGupta-merged", "EricMead_2009P_EricMead-merged"]:
    y, _ = sf.read(f"{ROOT}/data/ted/{t}.wav", dtype="float32"); dur = len(y) / 16000
    P = np.load(f"{ROOT}/probs/ted_{t.replace('-merged','')}.silero.npy")
    sil = P < 0.3; s, e = vr.runs(sil); mids = [(a + b) / 2 * 0.032 for a, b in zip(s, e) if (b - a) * 0.032 >= 0.4]
    rms = np.sqrt((y ** 2).mean())
    ins = []
    for frac, key, sec in ((0.25, "fma", 40), (0.5, "noise", 30), (0.75, "jamendo", 40)):
        c = clip(pick[key], sec); c = c * (0.7 * rms / (np.sqrt((c ** 2).mean()) + 1e-9)); c = np.clip(c, -1, 1)
        fade = np.linspace(0, 1, 1600, dtype=np.float32); c[:1600] *= fade; c[-1600:] *= fade[::-1]
        pos = min(mids, key=lambda m: abs(m - frac * dur)); ins.append((pos, c))
    out = []; last = 0.0
    for pos, c in sorted(ins, key=lambda x: x[0]):
        out += [y[int(last * 16000):int(pos * 16000)], c]; last = pos
    out.append(y[int(last * 16000):]); z = np.concatenate(out)
    name = t.replace("-merged", "") + "+ins"
    sf.write(f"{ROOT}/data/comp/{name}.wav", z, 16000, subtype="PCM_16")
    open(f"{ROOT}/data/comp/{name}.txt", "w").write(open(f"{ROOT}/data/ted/{t}.txt").read())
    print(name, round(len(z) / 16000), "s", [round(p) for p, _ in ins])
