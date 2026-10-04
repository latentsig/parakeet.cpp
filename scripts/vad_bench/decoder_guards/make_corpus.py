#!/usr/bin/env python3
"""usage: make_corpus.py DATA

Builds, from the output of fetch_data.py, the files of the runs (all 16 kHz mono, > 30 s so that the
segmenter cuts them) and DATA/manifest.json:
  talk_*       the talks as they are (kind "talk", reference text)
  ins_<talk>_<noise><rel>  90 s of a talk with 60 s of synthetic noise inserted at a quiet point
                           (kind "insert", "ins": [start, end] of the noise block). The noise level is
                           `rel` dB against the RMS of the speech. Noise: white, pink, clicks, music-like tones.
  nz_<talk>_<noise><rel>  30 s of noise alone (white, pink, clicks, music, hum, tone, sweep), no speech
                           (kind "noise"; shorter than the 30 s cap, so decoded whole)
  sn_<cond>_<k>  6 LibriSpeech utterances with gaps (kind "sinr"), clean or with white or pink noise at an SNR
                           (white5 = white noise at 5 dB, pink0 = pink noise at 0 dB)
Seeds are fixed, so the files are the same on every run."""
import glob, json, os, sys
import numpy as np, soundfile as sf

D = sys.argv[1]; SR = 16000
man = []

def rms(x): return float(np.sqrt((x ** 2).mean()) + 1e-12)
def at_db(x, db): return x * (10 ** (db / 20) / rms(x))
def pink(n, r):
    X = np.fft.rfft(r.standard_normal(n)); f = np.arange(len(X)); f[0] = 1
    y = np.fft.irfft(X / np.sqrt(f), n); return y / y.std()
def clicks(n, r):
    x = np.zeros(n); t = 0
    while t < n - 400:
        k = np.arange(400); x[t:t + 400] += r.uniform(0.3, 1) * r.choice([-1, 1]) * np.exp(-k / 40) * np.sin(k * 0.8)
        t += int(r.uniform(0.15, 0.5) * SR)
    return x
def music(n, r):
    t = np.arange(n) / SR; x = np.zeros(n)
    for f0 in (220.0, 261.63, 329.63, 440.0):
        f = f0 * (1 + 0.003 * np.sin(2 * np.pi * r.uniform(4, 6) * t + r.uniform(0, 6)))
        ph = 2 * np.pi * np.cumsum(f) / SR
        for h, a in ((1, 1.0), (2, 0.5), (3, 0.25)): x += a * np.sin(h * ph)
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t + r.uniform(0, 6)) ** 2
    return x * env
def hum(n, r):
    t = np.arange(n) / SR; f = r.choice([50.0, 60.0])
    return sum(np.sin(2 * np.pi * f * h * t + r.uniform(0, 6)) / h for h in (1, 2, 3, 5))
def tone(n, r): return np.sin(2 * np.pi * float(r.choice([440, 1000, 2000])) * np.arange(n) / SR)
def sweep(n, r):
    t = np.arange(n) / SR; T = t[-1]; return np.sin(2 * np.pi * (50 * t + (7500 - 50) * t * t / (2 * T)))
GEN = {"white": lambda n, r: r.standard_normal(n), "pink": pink, "clicks": clicks, "music": music,
       "hum": hum, "tone": tone, "sweep": sweep}

# talks and inserts
for wav in sorted(glob.glob(f"{D}/talk_*.wav")):
    name = os.path.basename(wav)[5:-4]
    x, sr = sf.read(wav, dtype="float32"); assert sr == SR
    man.append(dict(name=f"talk_{name}", kind="talk", talk=name, ref=open(wav[:-4] + ".txt").read().strip()))
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
        man.append(dict(name=nm, kind="insert", talk=name, noise=tn, level=rel, ins=[cut / SR, cut / SR + 60]))

    # 30 s of noise alone, at the same levels against the speech (decoded whole, no cuts)
    for ni, tn in enumerate(["white", "pink", "clicks", "music", "hum", "tone", "sweep"]):
        for rel in (-35, -20, -5):
            r = np.random.default_rng(5000 + 100 * ni + abs(rel) + len(name))
            nm = f"nz_{name}_{tn}{rel}"
            sf.write(f"{D}/{nm}.wav", np.clip(at_db(GEN[tn](30 * SR, r), db + rel), -1, 1).astype(np.float32), SR, subtype="PCM_16")
            man.append(dict(name=nm, kind="noise", talk=name, noise=tn, level=rel))

# speech in noise from LibriSpeech
U = [u.astype(np.float32) / 32768 for u in np.load(f"{D}/libri.npy", allow_pickle=True)]
T = json.load(open(f"{D}/libri.json"))
def trim(u):
    fr = len(u) // 160; r = np.sqrt((u[:fr * 160].reshape(fr, 160) ** 2).mean(1))
    idx = np.where(r > r.max() * 0.01)[0]; return u[idx[0] * 160:(idx[-1] + 1) * 160]
for k in range(len(U) // 6):
    ids = list(range(6 * k, 6 * k + 6)); r = np.random.default_rng(2000 + k)
    parts, spans, t = [r.standard_normal(int(r.uniform(.5, 1.5) * SR)) * 1e-3], [], 0.0
    t = len(parts[0]) / SR
    for j, i in enumerate(ids):
        u = trim(U[i]); spans.append((t, t + len(u) / SR)); parts.append(u); t += len(u) / SR
        g = r.standard_normal(int(r.uniform(.3, 2.0) * SR)) * 1e-3; parts.append(g); t += len(g) / SR
    y = np.concatenate(parts).astype(np.float32)
    sp = np.concatenate([y[int(s * SR):int(e * SR)] for s, e in spans]); P = (sp ** 2).mean()
    for cond, kind, snr in (("clean", None, None), ("white5", "white", 5), ("pink0", "pink", 0)):
        z = y
        if kind:
            rr = np.random.default_rng(3000 + k)
            n = rr.standard_normal(len(y)) if kind == "white" else pink(len(y), rr)
            z = np.clip(y + n * np.sqrt(P / 10 ** (snr / 10)), -1, 1).astype(np.float32)
        nm = f"sn_{cond}_{k:02d}"; sf.write(f"{D}/{nm}.wav", z, SR, subtype="PCM_16")
        man.append(dict(name=nm, kind="sinr", cond=cond, ref=" ".join(T[i] for i in ids)))
json.dump(man, open(f"{D}/manifest.json", "w"))
from collections import Counter
print(len(man), Counter(m["kind"] for m in man))
