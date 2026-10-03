#!/usr/bin/env python3
"""Write the onnxruntime reference probabilities used by tests/test_silero_vad.cpp.

The input clip is built from an existing fixture, not stored: 1.0 s of digital
silence, tests/fixtures/speech.wav, 1.5 s of digital silence (16 kHz). The 8 kHz
clip keeps every second sample of it. The last chunk is zero-padded to a full
chunk, as in the official audio_forward(). The clip length is not a multiple of
the chunk size on purpose, so the test also covers the padded final chunk.

Run the official model chunk by chunk, with the 64 (16 kHz) or 32 (8 kHz)
sample context, and print one line "<rate> <probability>" per chunk:

    pip install onnxruntime numpy
    python scripts/gen_silero_vad_ref.py silero_vad.onnx > tests/fixtures/silero_vad_ref.txt

The ONNX file is silero-vad v6.2.3 (MIT), src/silero_vad/data/silero_vad.onnx.
"""
import hashlib
import pathlib
import sys
import wave

import numpy as np
import onnxruntime as ort

ROOT = pathlib.Path(__file__).resolve().parent.parent


def build_clip():
    with wave.open(str(ROOT / "tests/fixtures/speech.wav")) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        sp = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
    return np.concatenate([np.zeros(16000, np.float32), sp, np.zeros(24000, np.float32)])


def run(sess, x, sr):
    chunk, ctx = (512, 64) if sr == 16000 else (256, 32)
    x = np.concatenate([x, np.zeros((-len(x)) % chunk, np.float32)])
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros((1, ctx), np.float32)
    out = []
    for i in range(0, len(x), chunk):
        inp = np.concatenate([context, x[None, i : i + chunk]], 1)
        p, state = sess.run(None, {"input": inp, "state": state, "sr": np.array(sr, dtype="int64")})
        out.append(float(p[0, 0]))
        context = inp[:, -ctx:]
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    onnx_path = pathlib.Path(sys.argv[1])
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    clip = build_clip()
    print("# Silero VAD reference probabilities from onnxruntime %s" % ort.__version__)
    print("# model sha256 %s" % hashlib.sha256(onnx_path.read_bytes()).hexdigest())
    print("# clip: 16000 zeros + tests/fixtures/speech.wav + 24000 zeros; 8 kHz = every second sample")
    for sr, x in ((16000, clip), (8000, clip[::2].copy())):
        for p in run(sess, x, sr):
            print("%d %.7g" % (sr, p))


if __name__ == "__main__":
    main()
