#!/usr/bin/env python3
"""Build the sets. usage: mkcorpus.py DATA
 talks:   tune = JamesCameron, JaneMcGonigal, DanBarber, RobertGupta; heldout = the others (>=30 s)
 sn_*:    LibriSpeech test-clean (every 13th utt; 40 speakers) split by SPEAKER (even index in sorted speaker list = tune, odd = heldout);
          layouts of 6 utterances with 0.3-2.0 s gaps (the PR's construction); each layout in 3 noise seeds x
          conditions clean, white{20,10,5,0}, pink{20,10,5,0} (SNR against the speech power).
          heldout: 10 layouts (60 utts), tune: 4 layouts (24 utts).
 ins_*:   90 s of a talk with 60 s of noise inserted at a quiet point (the PR's construction)."""
import glob, json, os, sys
import numpy as np, soundfile as sf
D = sys.argv[1]; SR = 16000
sys.path.insert(0, os.path.dirname(__file__))
from gens import GEN, pink, rms, at_db
TUNE_TALKS = {"JamesCameron-merged", "JaneMcGonigal-merged", "DanBarber-merged", "RobertGupta-merged"}
man = []
for wav in sorted(glob.glob(f"{D}/talk_*.wav")):
    name = os.path.basename(wav)[5:-4]
    x, sr = sf.read(wav, dtype="float32")
    if len(x) / SR < 60: continue
    split = "tune" if name in TUNE_TALKS else "heldout"
    man.append(dict(name=f"talk_{name}", kind="talk", split=split, ref=open(wav[:-4] + ".txt").read().strip(), dur=len(x) / SR))
    if len(x) < 150 * SR: continue
    ex = x[60 * SR:150 * SR]; db = 20 * np.log10(rms(ex))
    fr = len(ex) // 1600; e = 10 * np.log10((ex[:fr * 1600].reshape(fr, 1600) ** 2).mean(1) + 1e-12)
    c = [(e[i:i + 10].mean(), i) for i in range(250, 650)]; _, i0 = min(c); cut = (i0 + 5) * 1600
    for ti, (tn, rel) in enumerate([("white", -20), ("pink", -20), ("clicks", -20), ("music", -20), ("white", -35), ("music", -35),
                                   ("white", -5), ("pink", -5), ("clicks", -5), ("music", -5)]):
        r = np.random.default_rng(1000 + 17 * ti + len(name))
        n = np.clip(at_db(GEN[tn](60 * SR, r), db + rel), -1, 1)
        y = np.concatenate([ex[:cut], n, ex[cut:]]).astype(np.float32)
        nm = f"ins_{name}_{tn}{rel}"
        sf.write(f"{D}/{nm}.wav", y, SR, subtype="PCM_16")
        man.append(dict(name=nm, kind="insert", split=split, talk=name, noise=tn, level=rel, ins=[cut / SR, cut / SR + 60]))

U = [u.astype(np.float32) / 32768 for u in np.load(f"{D}/libri.npy", allow_pickle=True)]
L = json.load(open(f"{D}/libri.json")); T = L["text"]; SPK = L["speaker"]
spk = sorted(set(SPK)); tune_spk = set(spk[0::2])
def trim(u):
    fr = len(u) // 160; r = np.sqrt((u[:fr * 160].reshape(fr, 160) ** 2).mean(1))
    idx = np.where(r > r.max() * 0.01)[0]; return u[idx[0] * 160:(idx[-1] + 1) * 160]
rng = np.random.default_rng(7)
pools = {}
for split in ("tune", "heldout"):
    ids = [i for i in range(len(U)) if (SPK[i] in tune_spk) == (split == "tune")]
    rng.shuffle(ids); pools[split] = ids
NLAY = {"tune": 5, "heldout": 12}
CONDS = [("clean", None, None)] + [(f"white{s}", "white", s) for s in (20, 10, 5, 0)] + [(f"pink{s}", "pink", s) for s in (20, 10, 5, 0)]
for split in ("tune", "heldout"):
    for k in range(NLAY[split]):
        ids = pools[split][6 * k:6 * k + 6]; r = np.random.default_rng(2000 + k + (500 if split == "tune" else 0))
        parts = [r.standard_normal(int(r.uniform(.5, 1.5) * SR)) * 1e-3]; spans = []; t = len(parts[0]) / SR
        for i in ids:
            u = trim(U[i]); spans.append((t, t + len(u) / SR)); parts.append(u); t += len(u) / SR
            g = r.standard_normal(int(r.uniform(.3, 2.0) * SR)) * 1e-3; parts.append(g); t += len(g) / SR
        y = np.concatenate(parts).astype(np.float32)
        sp = np.concatenate([y[int(s * SR):int(e * SR)] for s, e in spans]); P = (sp ** 2).mean()
        for cond, kind, snr in CONDS:
            for seed in (range(3) if kind else range(1)):
                z = y
                if kind:
                    rr = np.random.default_rng(3000 + 10 * k + seed + (777 if split == "tune" else 0))
                    n = rr.standard_normal(len(y)) if kind == "white" else pink(len(y), rr)
                    z = np.clip(y + n * np.sqrt(P / 10 ** (snr / 10)), -1, 1).astype(np.float32)
                nm = f"sn_{split}_{cond}_L{k:02d}s{seed}"; sf.write(f"{D}/{nm}.wav", z, SR, subtype="PCM_16")
                man.append(dict(name=nm, kind="sinr", split=split, cond=cond, layout=k, seed=seed, dur=len(y) / SR,
                                spans=spans, utts=[T[i] for i in ids], speakers=[SPK[i] for i in ids], ref=" ".join(T[i] for i in ids)))
json.dump(man, open(f"{D}/manifest.json", "w"))
from collections import Counter
print(len(man), Counter((m["kind"], m["split"]) for m in man))
