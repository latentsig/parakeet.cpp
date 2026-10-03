#!/usr/bin/env python3
"""Compare two directories of bench-style hypothesis JSONs (<dir>/<lang>.json).

Prints, per language, how many utterances are identical after normalize() and the
WER of A scored against B as the reference (A = parakeet.cpp, B = HF reference).

  python3 scripts/compare_hyps.py ours_dir hf_dir
"""
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from asr_metrics import _edit_distance, normalize  # noqa: E402

def clean(t):
    # The HF side prints the <unk> piece where parakeet.cpp drops it; ignore that
    # one decoding-detail difference so it does not count as an ASR difference.
    return re.sub(r"<unk>", " ", t)


a, b = map(pathlib.Path, sys.argv[1:3])
tot_e = tot_w = tot_same = tot_n = 0
for ja in sorted(a.glob("*.json")):
    jb = b / ja.name
    if not jb.exists():
        continue
    fa = {f["path"]: f["text"] for f in json.loads(ja.read_text())["files"]}
    fb = {f["path"]: f["text"] for f in json.loads(jb.read_text())["files"]}
    e = w = same = 0
    for p in fb:
        x, y = normalize(clean(fa[p])).split(), normalize(clean(fb[p])).split()
        e += _edit_distance(y, x)
        w += len(y)
        same += x == y
        if x != y:
            print(f"  DIFF {ja.stem} {pathlib.Path(p).name}\n    ours: {fa[p]}\n    hf:   {fb[p]}")
    print(f"{ja.stem}: identical {same}/{len(fb)}  WER(ours vs hf) {100.0 * e / w:.2f}%  ({e}/{w} words)")
    tot_e += e; tot_w += w; tot_same += same; tot_n += len(fb)
print(f"TOTAL: identical {tot_same}/{tot_n}  WER(ours vs hf) {100.0 * tot_e / tot_w:.2f}%  ({tot_e}/{tot_w} words)")
