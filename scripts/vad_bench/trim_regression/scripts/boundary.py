"""Where do the errors change between two variants? usage: boundary.py DET VA VB [split] [kinds]
Zones are defined from VB's trimmed segments (the 'narrow' ones): cutaway = inside VA's segment but outside VB's;
start/end edge = first/last EDGE s inside VB's segment; interior = the rest. Time of an error: word mid for S/I,
for D the position inside the gap between the neighbouring hypothesis words."""
import sys, collections; sys.path.insert(0, __import__('os').path.dirname(__file__))
from groups import *
EDGE = 0.5
def events_with_time(det, name, vn):
    r = file_errors(det, name, vn); segs = segs_for(det, name, vn); hy = hyp(det, name, segs)
    ev = r["events"]; out = []
    # hypothesis neighbours for deletions: need the full alignment order -> recompute positions from the event list
    full = align(norm(MAN[name]["ref"]), hy)
    prev_end = 0.0; pend = []; hmax = None
    def flush(next_start):
        nonlocal pend
        k = len(pend)
        for i, (j) in enumerate(pend):
            out.append(("D", prev_end + (next_start - prev_end) * (i + 1) / (k + 1), j, None))
        pend = []
    for kind, rj, hj in full:
        if kind == "D": pend.append(rj); continue
        t0, t1 = hy[hj][1], hy[hj][2]
        if pend: flush(t0)
        if kind in "SI": out.append((kind, (t0 + t1) / 2, rj, hj))
        prev_end = t1
    if pend: flush(prev_end + 0.5)
    return out, hy, full, segs
def zone(t, segsB):
    for i, (s, e) in enumerate(segsB):
        if s <= t <= e:
            if t < s + EDGE: return "start-edge"
            if t > e - EDGE: return "end-edge"
            return "interior"
    return "cutaway"
if __name__ == "__main__":
    det, va, vb = sys.argv[1:4]; split = sys.argv[4] if len(sys.argv) > 4 else "all"; kinds = sys.argv[5].split(",") if len(sys.argv) > 5 else ["talk", "sinr"]
    nms = [n for k in kinds for n in names(k, None if split == "all" else split)]
    tab = collections.defaultdict(lambda: [0, 0]); clipped = collections.Counter()
    for nm in nms:
        try: ea, hya, fa, sa = events_with_time(det, nm, va); eb, hyb, fb, sb = events_with_time(det, nm, vb)
        except KeyError: continue
        for kind, t, rj, hj in ea: tab[(zone(t, sb), kind)][0] += 1
        for kind, t, rj, hj in eb: tab[(zone(t, sb), kind)][1] += 1
        # status of VA hypothesis tokens that VB's segments cut away
        stat = {hj: kind for kind, rj, hj in fa if hj is not None}
        for hj, (tok, st, en, si, cf) in enumerate(hya):
            mid = (st + en) / 2
            if zone(mid, sb) == "cutaway": clipped["tokens in cut-away"] += 1; clipped["  of which correct(C)" if stat[hj] == "C" else "  of which error (S/I)"] += 1
        clipped["tokens total (A)"] += len(hya)
    print(f"== {det} {va} -> {vb}  split={split} kinds={kinds}   (errors by zone: A / B / B-A)")
    for z in ("cutaway", "start-edge", "end-edge", "interior"):
        row = []
        for k in "SDI":
            a, b = tab[(z, k)]; row.append(f"{k}: {a:4d}/{b:4d}/{b-a:+4d}")
        a = sum(tab[(z, k)][0] for k in "SDI"); b = sum(tab[(z, k)][1] for k in "SDI")
        print(f"  {z:11s} " + "  ".join(row) + f"   total {a:4d}/{b:4d}/{b-a:+4d}")
    print("  ", dict(clipped))
