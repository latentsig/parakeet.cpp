#!/usr/bin/env python3
"""Build the synthetic clips of the Silero parity benchmark.

usage: make_clips.py DATA_DIR     (DATA_DIR holds libri.npy from fetch_data.py)

Writes DATA_DIR/clips/*.wav and DATA_DIR/clips.json: 40 clips of 5 utterances in 3 conditions
(clean, white noise at 10 dB SNR, pink noise at 10 dB SNR) = 120 clips. The reference speech
spans are the utterance spans (trimmed at 1 percent of peak frame RMS), so they are coarse.
"""
import json, os, sys
import numpy as np, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vad_synth import build, add_noise

D = sys.argv[1]
os.makedirs(D + "/clips", exist_ok=True)
libri = list(np.load(D + "/libri.npy", allow_pickle=True))
rng = np.random.default_rng(7)
order = rng.permutation(len(libri))
sets = [[libri[i] for i in order[k * 5:(k + 1) * 5]] for k in range(len(libri) // 5)]
conds = [("clean", None), ("white", 10), ("pink", 10)]
meta = []
for ci, (kind, snr) in enumerate(conds):
    for k, us in enumerate(sets):
        y, spans = build(us, np.random.default_rng(1000 + k))
        y = add_noise(y, spans, kind, snr, np.random.default_rng(5000 + ci * 100 + k))
        name = f"{kind}{'' if snr is None else snr}_{k:02d}"
        sf.write(f"{D}/clips/{name}.wav", y, 16000, subtype="PCM_16")
        meta.append(dict(name=name, cond=name.split("_")[0], ref=spans, dur=len(y) / 16000))
json.dump(meta, open(D + "/clips.json", "w"))
print(len(meta), "clips", sum(m["dur"] for m in meta), "s")
