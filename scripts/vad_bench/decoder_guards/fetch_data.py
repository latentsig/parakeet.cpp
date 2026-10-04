#!/usr/bin/env python3
"""usage: fetch_data.py OUT [--skip K] [--talks N] [--libri N]

Streams the data of the trim and word filter runs; nothing is kept but what is written to OUT:
  OUT/talk_<name>.wav, talk_<name>.txt   TED-LIUM long-form (test), talks under 1500 s, after the
                                         first K talks (default 4: the talks that were used to tune
                                         things earlier are skipped), N talks (default 3)
  OUT/libri.npy, libri.json              every 13th LibriSpeech test-clean utterance, first N (default 60)
Delete OUT after the runs: it is audio."""
import argparse, io, json, os, re
import numpy as np, soundfile as sf, librosa
from datasets import Audio, load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--skip", type=int, default=4)
ap.add_argument("--talks", type=int, default=3); ap.add_argument("--libri", type=int, default=60)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

ds = load_dataset("distil-whisper/tedlium-long-form", split="test", streaming=True).cast_column("audio", Audio(decode=False))
kept = 0
for i, ex in enumerate(ds):
    if i < a.skip: continue
    y, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    if len(y) / 16000 > 1500: continue
    name = re.sub(r"[^A-Za-z0-9_-]", "", os.path.basename(ex["audio"]["path"] or f"talk{i}").rsplit(".", 1)[0]) or f"talk{i}"
    sf.write(f"{a.out}/talk_{name}.wav", y, 16000, subtype="PCM_16")
    open(f"{a.out}/talk_{name}.txt", "w").write(" ".join(re.sub(r"<[^>]*>", " ", ex["text"]).split()) + "\n")
    print("talk", name, round(len(y) / 16000), flush=True); kept += 1
    if kept >= a.talks: break

ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True).cast_column("audio", Audio(decode=False))
U, T = [], []
for i, ex in enumerate(ds):
    if i % 13: continue
    y, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="int16"); assert sr == 16000
    U.append(y); T.append(ex["text"].lower())
    if len(U) >= a.libri: break
np.save(f"{a.out}/libri.npy", np.array(U, dtype=object), allow_pickle=True)
json.dump(T, open(f"{a.out}/libri.json", "w"))
print("libri", len(U), flush=True)
