#!/usr/bin/env python3
"""Corpus WER of one or more GGUF models over per-language manifests.

Each manifest is a TSV of `wav_path<TAB>reference`.  For every (model, manifest)
pair this runs `parakeet-cli bench --json`, keeps the hypotheses in
<out>/<model-name>/<manifest-dir-name>.json (reused if present, so reruns are
cheap) and scores with scripts/asr_metrics.normalize on BOTH sides.  That is not
the Open ASR Leaderboard normalizer, so absolute numbers are not comparable with
published model cards; compare models against each other on the same subset.

WER per manifest is a corpus WER (total edits / total reference words).  The
summary line is the unweighted mean over manifests (macro average).

  python3 scripts/eval_manifest_wer.py --out /tmp/val/res \
      --model v3=models/v3.gguf --model ultra=models/ultra.gguf \
      /tmp/val/fleurs/en_us/manifest.tsv /tmp/val/fleurs/de_de/manifest.tsv
"""
import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from asr_metrics import _edit_distance, normalize  # noqa: E402


def score(manifest, doc):
    refs = {}
    for line in pathlib.Path(manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p, r = line.split("\t", 1)
            refs[p] = r
    edits = words = 0
    for f in doc["files"]:
        r = normalize(refs[f["path"]]).split()
        h = normalize(f["text"]).split()
        edits += _edit_distance(r, h)
        words += len(r)
    audio = sum(f["audio_sec"] for f in doc["files"])
    proc = sum(f["proc_ms"] for f in doc["files"]) / 1000.0
    return edits, words, audio, proc, len(doc["files"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="+")
    ap.add_argument("--model", action="append", required=True, help="name=path.gguf (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cli", default="build/examples/cli/parakeet-cli")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--decoder", default="tdt")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    table = {}
    for spec in a.model:
        name, path = spec.split("=", 1)
        (out / name).mkdir(parents=True, exist_ok=True)
        for m in a.manifests:
            lang = pathlib.Path(m).parent.name
            js = out / name / f"{lang}.json"
            if not js.exists():
                subprocess.run([a.cli, "bench", "--model", path, "--manifest", m, "--decoder",
                                a.decoder, "--threads", str(a.threads), "--json", str(js)],
                               check=True, stdout=subprocess.DEVNULL)
            e, w, au, pr, n = score(m, json.loads(js.read_text()))
            table[(name, lang)] = (100.0 * e / w, w, au, n)
    names = [s.split("=", 1)[0] for s in a.model]
    langs = sorted({l for _, l in table})
    print("lang\t" + "\t".join(names))
    for l in langs:
        print(l + "\t" + "\t".join(f"{table[(n, l)][0]:.2f}" for n in names))
    print("MEAN\t" + "\t".join(f"{sum(table[(n, l)][0] for l in langs) / len(langs):.2f}" for n in names))
    n0 = names[0]
    print(f"utterances={sum(table[(n0, l)][3] for l in langs)} "
          f"ref_words={sum(table[(n0, l)][1] for l in langs)} "
          f"audio_sec={sum(table[(n0, l)][2] for l in langs):.1f}")


if __name__ == "__main__":
    main()
