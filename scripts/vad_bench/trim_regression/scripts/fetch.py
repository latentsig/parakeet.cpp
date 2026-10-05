#!/usr/bin/env python3
"""Stream TED-LIUM long-form test talks (<=1500 s, all of them) and LibriSpeech test-clean (every 13th utt, with speaker ids)."""
import io, json, os, re, sys
import numpy as np, soundfile as sf, librosa
from datasets import Audio, load_dataset
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
os.environ["HF_HOME"] = os.path.abspath(os.path.join(out, "..", "hfhome"))
ds = load_dataset("distil-whisper/tedlium-long-form", split="test", streaming=True).cast_column("audio", Audio(decode=False))
for i, ex in enumerate(ds):
    y, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != 16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    name = re.sub(r"[^A-Za-z0-9_-]", "", os.path.basename(ex["audio"]["path"] or f"talk{i}").rsplit(".", 1)[0]) or f"talk{i}"
    print("talk", i, name, round(len(y) / 16000), flush=True)
    if len(y) / 16000 > 1500: continue
    sf.write(f"{out}/talk_{name}.wav", y, 16000, subtype="PCM_16")
    open(f"{out}/talk_{name}.txt", "w").write(" ".join(re.sub(r"<[^>]*>", " ", ex["text"]).split()) + "\n")
ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True).cast_column("audio", Audio(decode=False))
U, T, SP = [], [], []
for i, ex in enumerate(ds):
    if i % 13: continue
    y, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="int16"); assert sr == 16000
    U.append(y); T.append(ex["text"].lower()); SP.append(int(ex["speaker_id"]))
np.save(f"{out}/libri.npy", np.array(U, dtype=object), allow_pickle=True)
json.dump({"text": T, "speaker": SP}, open(f"{out}/libri.json", "w"))
print("libri", len(U), len(set(SP)), flush=True)
