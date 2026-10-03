#!/usr/bin/env python3
"""Transcribe manifests with the transformers ParakeetForTDT reference (CPU).

Used to check parakeet.cpp against an independent implementation of a Parakeet
derivative that only ships in HF format (e.g. moondream/parakeet-ultra).  The
HF checkpoint has no preprocessor or tokenizer files, so this builds a
ParakeetFeatureExtractor with the checkpoint's mel count and decodes token ids
with the sentencepiece piece table stored in a parakeet.cpp GGUF of the same
model (`parakeet.tokenizer.pieces`; a leading U+2581 is a word start).

ParakeetForTDT is not in every transformers release; point PYTHONPATH at a
transformers source tree that has it if the installed one does not.

  PYTHONPATH=<transformers>/src python3 scripts/hf_reference_transcribe.py \
      --hf-dir <hf-checkpoint-dir> --gguf ultra-f16.gguf --out hyp/ en_us/manifest.tsv ...

Writes <out>/<manifest-dir-name>.json in the same {"files":[{path,text}]} shape
as `parakeet-cli bench --json`, so scripts/eval_manifest_wer.py can score it.
"""
import argparse
import json
import pathlib

import gguf
import numpy as np
import soundfile as sf
import torch
from transformers import ParakeetFeatureExtractor, ParakeetForTDT


def pieces_from_gguf(path):
    f = gguf.GGUFReader(path).fields["parakeet.tokenizer.pieces"]
    return [bytes(f.parts[i]).decode("utf-8") for i in f.data]


def detok(ids, pieces, blank):
    return "".join(pieces[i] for i in ids if i != blank and i < len(pieces)).replace("▁", " ").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = ParakeetForTDT.from_pretrained(a.hf_dir, torch_dtype=torch.float32).eval()
    fe = ParakeetFeatureExtractor(feature_size=model.config.encoder_config.num_mel_bins, sampling_rate=16000)
    pieces = pieces_from_gguf(a.gguf)
    blank = model.config.blank_token_id
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for m in a.manifests:
        files = []
        for line in pathlib.Path(m).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = line.split("\t", 1)[0]
            y, sr = sf.read(p, dtype="float32")
            assert sr == 16000
            feats = fe(y, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            with torch.no_grad():
                seq = model.generate(**feats, decoder_start_token_id=blank,
                                     suppress_tokens=list(range(model.config.vocab_size, model.config.vocab_size + len(model.config.durations))))
            seq = seq.sequences if hasattr(seq, "sequences") else seq
            files.append({"path": p, "text": detok(seq[0].tolist(), pieces, blank), "audio_sec": len(y) / 16000, "proc_ms": 0})
        (out / f"{pathlib.Path(m).parent.name}.json").write_text(json.dumps({"files": files}, ensure_ascii=False))
        print(m, len(files), flush=True)


if __name__ == "__main__":
    main()
