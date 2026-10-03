#!/usr/bin/env python3
"""Fetch the public audio used by the VAD benchmarks (streams from the Hugging Face hub).

usage: fetch_data.py OUT_DIR [--libri-count N] [--ted-seconds S]

Writes OUT_DIR/libri.npy (LibriSpeech test-clean, every 13th utterance, N of them) and
OUT_DIR/ted_talk.wav (the first TED-LIUM long-form talk longer than 700 s, cut to S seconds,
default 600). The silero parity run used N=200. The vad-compare run used a subset with 24 x 5
utterances; the script that cut it was not saved, so that corpus is not byte-identical to the
one in the published tables (see docs/vad-benchmarks.md).
"""
import argparse, io, os
import numpy as np, soundfile as sf, librosa
from datasets import load_dataset, Audio

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--libri-count", type=int, default=200)
ap.add_argument("--ted-seconds", type=int, default=600)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True).cast_column("audio", Audio(decode=False))
U = []
for i, ex in enumerate(ds):
    if i % 13:
        continue
    y, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
    assert sr == 16000
    U.append(y)
    if len(U) >= a.libri_count:
        break
np.save(f"{a.out}/libri.npy", np.array(U, dtype=object), allow_pickle=True)
print("libri", len(U))

ds = load_dataset("distil-whisper/tedlium-long-form", split="test", streaming=True).cast_column("audio", Audio(decode=False))
for ex in ds:
    y, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
    if y.ndim > 1:
        y = y.mean(1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    if len(y) / 16000 < 700:
        continue
    sf.write(f"{a.out}/ted_talk.wav", y[: a.ted_seconds * 16000], 16000, subtype="PCM_16")
    print("ted", len(y) / 16000)
    break
