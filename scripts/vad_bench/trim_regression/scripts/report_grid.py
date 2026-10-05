"""usage: report_grid.py DET SPLIT VARIANTS [BASE=T0]   rows = variants; columns = groups. WER and paired delta vs BASE with 95% bootstrap CI (* = CI excludes 0)."""
import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from groups import *
det, split, vs = sys.argv[1], sys.argv[2], sys.argv[3].split(","); base = sys.argv[4] if len(sys.argv) > 4 else "T0"
cols = [f"talks[{split}]", f"clean[{split}]", f"white[{split}]", f"pink[{split}]", f"pink0[{split}]", f"white0[{split}]", f"noisy-all[{split}]", f"sinr-all[{split}]"]
gs = G(split)
print(f"### {det}, {split}: WER % (delta vs {base}, paired bootstrap 95% CI)")
print("| variant | " + " | ".join(c.replace(f"[{split}]", "") for c in cols) + " |")
print("|---|" + "---|" * len(cols))
nrefs = {}
for v in [base] + [x for x in vs if x != base]:
    cells = []
    for c in cols:
        try:
            A, B, sa, sb = paired(det, base, v, gs[c]); w = boot_ci(B)[0]
            if v == base: cells.append(f"{w:.2f}")
            else:
                pt, lo, hi = boot_diff(A, B); cells.append(f"{w:.2f} ({pt:+.2f}; {lo:+.2f}..{hi:+.2f}){'*' if lo > 0 or hi < 0 else ''}")
            nrefs[c] = int(sum(x[1] for x in A))
        except (KeyError, FileNotFoundError): cells.append("n/a")
    print(f"| {v} | " + " | ".join(cells) + " |")
print("| words in reference | " + " | ".join(str(nrefs.get(c, "")) for c in cols) + " |")
