#!/usr/bin/env python3
"""Join LibriSpeech benchmark utterances into long clips with a known reference.

Each output clip is N utterances separated by a gap of low-level Gaussian noise
(about -55 dBFS, fixed seed) rather than digital zeros: digital silence distorts
the per-feature mel normalization and would not resemble real recordings.
Clips are written as <out>/longform_<gap>_<i>.wav with a matching .txt holding
the joined reference. A gap of 0 joins the utterances with no inserted audio.
"""
import argparse
import pathlib
import wave

import numpy as np

NOISE_STD = 58.0  # int16 units, about -55 dBFS


def read(path):
    with wave.open(path) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="benchmarks/librispeech_manifest.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-clip", type=int, default=30)
    ap.add_argument("--gap", type=float, default=0.45, help="noise gap between utterances, seconds (0 = none)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rows = []
    for line in open(args.manifest):
        if not line.strip() or line.startswith("#"):
            continue
        path, text = line.rstrip("\n").split("\t", 1)
        rows.append((path, text))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_gap = int(args.gap * 16000)
    tag = f"{args.gap:.2f}".replace(".", "p")
    for i in range(0, len(rows) - args.per_clip + 1, args.per_clip):
        chunk = rows[i:i + args.per_clip]
        parts = []
        for p, _ in chunk:
            parts.append(read(p))
            if n_gap:
                parts.append(np.clip(rng.normal(0, NOISE_STD, n_gap), -32768, 32767).astype(np.int16))
        audio = np.concatenate(parts)
        stem = out / f"longform_{tag}_{i // args.per_clip}"
        with wave.open(str(stem) + ".wav", "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(audio.tobytes())
        pathlib.Path(str(stem) + ".txt").write_text(" ".join(t for _, t in chunk) + "\n")
        print(f"{stem}.wav  {len(audio) / 16000:.1f} s  {len(chunk)} utterances")


if __name__ == "__main__":
    main()
