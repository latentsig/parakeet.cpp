"""usage: report_cand.py DET SPLIT VARIANTS BASE   columns: talks, clean, noisy (the SNR conditions listed), pink0, combined (talks+clean+noisy4)."""
import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from groups import *
det, split, vs, base = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
s = None if split == "all" else split
N4 = sys.argv[5].split(",") if len(sys.argv) > 5 else ["white5", "white0", "pink5", "pink0"]
def have(nm):
    out = []
    for n in nm:
        c = dec.load_cache(det, n)
        if all(dec.key(a, b) in c for v in [base] + vs for a, b in segs_for(det, n, v)): out.append(n)
    return out
cols = {"talks": names("talk", s), "clean": names("sinr", s, ["clean"]), "noisy": names("sinr", s, N4), "pink0": names("sinr", s, ["pink0"]), "white0": names("sinr", s, ["white0"]) if "white0" in N4 else []}
cols = {k: have(v) for k, v in cols.items()}
cols["combined"] = cols["talks"] + cols["clean"] + cols["noisy"]
print(f"### {det}, {split}: WER % (delta vs {base}; paired bootstrap 95% CI; * = CI excludes 0)")
print("| variant | " + " | ".join(cols) + " |"); print("|---|" + "---|" * len(cols))
nref = {}
for v in [base] + [x for x in vs if x != base]:
    cells = []
    for c, nm in cols.items():
        if not nm: cells.append("-"); continue
        try:
            A, B, sa, sb = paired(det, base, v, nm); w = boot_ci(B)[0]; nref[c] = int(sum(x[1] for x in A))
            if v == base: cells.append(f"{w:.2f}")
            else:
                pt, lo, hi = boot_diff(A, B); cells.append(f"{w:.2f} ({pt:+.2f}; {lo:+.2f}..{hi:+.2f}){'*' if lo > 0 or hi < 0 else ''}")
        except (KeyError, FileNotFoundError): cells.append("n/a")
    print(f"| {v} | " + " | ".join(cells) + " |")
print("| ref words | " + " | ".join(str(nref.get(c, "")) for c in cols) + " |")
