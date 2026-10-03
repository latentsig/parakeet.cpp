#!/usr/bin/env python3
"""Stream distil-whisper/tedlium-long-form (test split, 11 full talks) to disk.

Writes <out>/<talk>.wav (16 kHz mono int16) and <out>/<talk>.txt (the reference
with tags such as <unk> removed and whitespace collapsed).  Public, ungated, no
token needed.  Audio is streamed with decode=False and decoded with soundfile.

  python3 scripts/fetch_tedlium_longform.py --out /tmp/val/tedlium
"""
import argparse
import io
import pathlib
import re

import librosa
import soundfile as sf
from datasets import Audio, load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", help="write only the talk whose name contains this text (e.g. BillGates)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("distil-whisper/tedlium-long-form", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    for i, ex in enumerate(ds):
        name = re.sub(r"[^A-Za-z0-9_-]", "", pathlib.Path(ex["audio"]["path"] or f"talk{i}").stem) or f"talk{i}"
        if a.only and a.only not in name:
            continue
        y, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sf.write(str(out / f"{name}.wav"), y, 16000, subtype="PCM_16")
        text = " ".join(re.sub(r"<[^>]*>", " ", ex["text"]).split())
        (out / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
        print(f"{name}\t{len(y) / 16000:.1f} s\t{len(text.split())} words", flush=True)


if __name__ == "__main__":
    main()
