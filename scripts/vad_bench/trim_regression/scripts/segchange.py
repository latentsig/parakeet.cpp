"""Per segment: does the text change between two variants, and does the error count go up or down? (segments are 1:1 between variants.)
Errors are attributed to the segment of the hypothesis token (S, I) or, for D, of the preceding hypothesis token (first segment if none).
usage: segchange.py DET VA VB [kinds]"""
import sys, collections; sys.path.insert(0, __import__('os').path.dirname(__file__))
from groups import *
from scipy.stats import binomtest
def seg_errors(det, name, vn):
    segs = segs_for(det, name, vn); hy = hyp(det, name, segs); ev = align(norm(MAN[name]["ref"]), hy)
    E = np.zeros(len(segs), int); last = 0; texts = [[] for _ in segs]
    for t in hy: texts[t[3]].append(t[0])
    for kind, rj, hj in ev:
        if hj is not None: last = hy[hj][3]
        if kind in "SDI": E[last if kind == "D" else hy[hj][3]] += 1
    return E, [" ".join(t) for t in texts]
det, va, vb = sys.argv[1:4]; kinds = sys.argv[4].split(",") if len(sys.argv) > 4 else ["talk", "sinr"]
for kind in kinds:
    nms = names(kind) if kind == "talk" else names("sinr", None, W_ + P_)
    n = ch = up = down = 0; dsum = 0
    for nm in nms:
        try: Ea, Ta = seg_errors(det, nm, va); Eb, Tb = seg_errors(det, nm, vb)
        except KeyError: continue
        for i in range(len(Ea)):
            n += 1
            if Ta[i] != Tb[i]:
                ch += 1; d = Eb[i] - Ea[i]; dsum += d; up += d > 0; down += d < 0
    p = binomtest(up, up + down, 0.5).pvalue if up + down else 1.0
    print(f"{det:6s}{kind:5s} {va}->{vb}: segments {n}, text changed {ch} ({100*ch/n:.1f}%); of those errors up {up}, down {down}, equal {ch-up-down}; net {dsum:+d}; sign test p={p:.2f}")
