import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from groups import *
vs = sys.argv[1].split(","); split = sys.argv[2]; dets = sys.argv[3].split(",") if len(sys.argv) > 3 else list(DET)
base = vs[0]
for det in dets:
    print(f"\n#### {det}  split={split}  (WER %, delta vs {base} with paired bootstrap 95% CI; [S/D/I] counts)")
    for g, nm in G(split).items():
        try:
            row = []
            nref = None
            for v in vs:
                A, B, sa, sb = paired(det, base, v, nm)
                w = boot_ci(B)[0]
                if v == base: row.append(f"{w:5.2f} [{sa['S']}/{sa['D']}/{sa['I']}]")
                else:
                    pt, lo, hi = boot_diff(A, B)
                    sig = "*" if lo > 0 or hi < 0 else " "
                    row.append(f"{w:5.2f} {pt:+5.2f}({lo:+5.2f},{hi:+5.2f}){sig} [{sb['S']}/{sb['D']}/{sb['I']}]")
            nref = int(np.sum([x[1] for x in A]))
            print(f"{g:20s} n_ref={nref:6d} units={len(A):4d} | " + " | ".join(row))
        except Exception as ex:
            print(f"{g:20s} missing: {type(ex).__name__} {str(ex)[:60]}")
