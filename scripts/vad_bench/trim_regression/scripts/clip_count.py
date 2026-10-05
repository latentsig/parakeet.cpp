"""Words of the untrimmed (T0) decode that a variant's segments cut: counted from the T0 hypothesis and its alignment to the reference.
cut = the word interval overlaps the removed part by more than half (mid outside the kept segment) ; touched = any overlap with the removed part."""
import sys, collections; sys.path.insert(0, __import__('os').path.dirname(__file__))
from groups import *
vs = sys.argv[1].split(","); dets = sys.argv[2].split(",") if len(sys.argv) > 2 else list(DET)
sets = (("talks", names("talk")), ("sinr noisy", names("sinr", None, W_ + P_)), ("sinr clean", names("sinr", None, ["clean"])))
for det in dets:
    print(f"== {det}")
    for sname, nm in sets:
        base_tok = 0; rows = {v: collections.Counter() for v in vs}
        for n in nm:
            try:
                s0 = segs_for(det, n, "T0"); hy = hyp(det, n, s0); ev = align(norm(MAN[n]["ref"]), hy)
            except KeyError: continue
            st = {hj: k for k, rj, hj in ev if hj is not None}; base_tok += len(hy)
            for v in vs:
                sb = segs_for(det, n, v)
                for hj, (tok, s, e, si, cf) in enumerate(hy):
                    a, b = sb[si]   # same segment index (trim keeps the segment list)
                    out = max(0.0, a - s) + max(0.0, e - b)
                    if out <= 0: continue
                    kind = "correct" if st[hj] == "C" else "error"
                    rows[v]["touched " + kind] += 1
                    if out > 0.5 * (e - s): rows[v]["cut " + kind] += 1
        print(f"  {sname:11s} T0 words {base_tok}")
        for v in vs:
            c = rows[v]; print(f"     {v:12s} cut correct {c['cut correct']:4d}  cut error {c['cut error']:4d} | touched correct {c['touched correct']:4d} touched error {c['touched error']:4d}")
