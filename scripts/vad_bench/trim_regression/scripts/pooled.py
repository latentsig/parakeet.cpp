"""Pooled paired comparison across the three detectors (units = (det, utterance) for sinr; (det, talk block) for talks)."""
import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from groups import *
def pool(dets, va, vb, kind, conds=None, split=None):
    A = []; B = []
    for det in dets:
        nm = names("talk", split) if kind == "talk" else names("sinr", split, conds)
        nm = [n for n in nm if all(dec.key(a, b) in dec.load_cache(det, n) for v in (va, vb) for a, b in segs_for(det, n, v))]
        a, b, _, _ = paired(det, va, vb, nm); A += a; B += b
    return A, B
pairs = [("T0", "T0.3"), ("T0.3", "T0.5"), ("T0.3", "P0.5/0.3"), ("T0.3", "E0.5"), ("T0", "T0.5"), ("T0", "E0.5"), ("T0", "P0.5/0.3")]
N3 = ["white5", "pink5", "pink0"]
print("| comparison (B vs A) | set | words | WER A | WER B | delta (pp) | 95% CI |"); print("|---|---|---|---|---|---|---|")
for va, vb in pairs:
    for label, kind, conds, split in (("noisy sinr 3 dets, tune+heldout", "sinr", N3, None), ("talks 3 dets, all 9", "talk", None, None)):
        A, B = pool(["redux", "ultra", "v3"], va, vb, kind, conds, split)
        pt, lo, hi = boot_diff(A, B); wa = boot_ci(A)[0]; wb = boot_ci(B)[0]
        print(f"| {vb} vs {va} | {label} | {int(sum(x[1] for x in A))} | {wa:.2f} | {wb:.2f} | {pt:+.2f} | {lo:+.2f}..{hi:+.2f}{' *' if lo > 0 or hi < 0 else ''} |")
