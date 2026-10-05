"""Invented words in the 60 s noise block, on the 20 insert files that were decoded (4 talks x white-20, pink-5, music-20, clicks-5, white-5).
Only segments overlapping the block were decoded; words inside [block+0.5 s, block end-0.5 s] count (as in the PR)."""
import sys, re; sys.path.insert(0, __import__('os').path.dirname(__file__))
from lib import *
vs = sys.argv[1].split(","); dets = sys.argv[2].split(",")
INS = re.compile(r"ins_(JamesCameron|JaneMcGonigal|DanielKahneman|MichaelSpecter)-merged_(white-20|pink-5|music-20|clicks-5|white-5)$")
ins = [m for m in MAN.values() if m["kind"] == "insert" and INS.match(m["name"])]
def ov(segs, a, b): return sum(max(0.0, min(e, b) - max(s, a)) for s, e in segs)
print(f"{len(ins)} files, noise block 60 s each")
print("| detector | variant | noise sec decoded / file | invented words | files with words | words per file with noise decoded |")
print("|---|---|---|---|---|---|")
for det in dets:
    for v in vs:
        secs = []; wn = 0; fw = 0
        for m in ins:
            a, b = m["ins"]; segs = segs_for(det, m["name"], v); secs.append(ov(segs, a, b)); c = dec.load_cache(det, m["name"])
            k = 0
            for s, e in segs:
                if e <= a - 0.5 or s >= b + 0.5: continue
                k += sum(a + 0.5 <= w[0] <= b - 0.5 for w in c[dec.key(s, e)]["words"])
            wn += k; fw += k > 0
        print(f"| {det} | {v} | {np.mean(secs):.1f} | {wn} | {fw}/{len(ins)} | |")
