#!/usr/bin/env python3
"""Stream the first N test utterances of FLEURS languages into 16 kHz mono wavs.

For each FLEURS config id it writes <out>/<cfg>/NNNN.wav and <out>/<cfg>/manifest.tsv
(`wav_path<TAB>reference`, reference = the dataset's `raw_transcription`, the
natural-cased text with punctuation; scoring normalizes it).  Utterances are the
first N of the test split in dataset (parquet) order, no shuffling or filtering.

Only public, ungated data is used and no token is needed.  Audio is streamed with
decode=False and decoded here with soundfile, so torchcodec is not required.

  python3 scripts/fetch_fleurs_subset.py --out /tmp/val/fleurs --langs en_us,de_de --n 50
"""
import argparse
import io
import pathlib

import librosa
import soundfile as sf
from datasets import Audio, load_dataset

ALL = ("bg_bg hr_hr cs_cz da_dk nl_nl en_us et_ee fi_fi fr_fr de_de el_gr hu_hu it_it "
       "lv_lv lt_lt mt_mt pl_pl pt_br ro_ro ru_ru sk_sk sl_si es_419 sv_se uk_ua").split()


def fetch(cfg, out, n):
    d = pathlib.Path(out) / cfg
    d.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("google/fleurs", cfg, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    rows, secs = [], 0.0
    for i, ex in enumerate(ds):
        if i >= n:
            break
        y, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        p = d / f"{i:04d}.wav"
        sf.write(str(p), y, 16000, subtype="PCM_16")
        secs += len(y) / 16000
        ref = " ".join(ex["raw_transcription"].split())
        rows.append(f"{p}\t{ref}")
    (d / "manifest.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows), secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default=",".join(ALL))
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    for cfg in a.langs.split(","):
        k, s = fetch(cfg, a.out, a.n)
        print(f"{cfg}\t{k} utts\t{s:.1f} s", flush=True)


if __name__ == "__main__":
    main()
