"""Noise block of 60 s inside speech: seconds of the block that the decoder gets, and words invented in it (decode cache needed for words)."""
import sys, collections; sys.path.insert(0, __import__('os').path.dirname(__file__))
from lib import *
vs = sys.argv[1].split(","); dets = sys.argv[2].split(",") if len(sys.argv) > 2 else list(DET)
words = len(sys.argv) > 3 and sys.argv[3] == "words"
ins = [m for m in MAN.values() if m["kind"] == "insert"]
def ov(segs, a, b): return sum(max(0.0, min(e, b) - max(s, a)) for s, e in segs)
print(f"{len(ins)} insert files (9 talks x white,pink,clicks,music @-20; white,music @-35; white,pink,clicks,music @-5)")
print(f"{'det':6s}{'variant':10s}{'sec/file':>9s}" + ("{'words':>7s}{'files>0':>8s}" if words else ""))
for det in dets:
    for v in vs:
        secs = []; wn = 0; fw = 0; bytype = collections.defaultdict(list)
        for m in ins:
            a, b = m["ins"]; segs = segs_for(det, m["name"], v); s = ov(segs, a, b); secs.append(s)
            bytype[f"{m['noise']}{m['level']}"].append(s)
            if words:
                h = hyp(det, m["name"], segs); k = sum(a + 0.5 <= t[1] <= b - 0.5 for t in h); wn += k; fw += k > 0
        print(f"{det:6s}{v:10s}{np.mean(secs):9.1f}" + (f"{wn:7d}{fw:8d}" if words else ""), "  " + " ".join(f"{k}:{np.mean(x):.0f}" for k, x in sorted(bytype.items())))
