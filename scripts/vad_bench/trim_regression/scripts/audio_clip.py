"""Against the construction truth of the sinr files (utterance spans = first/last 10 ms frame above 1% of the peak RMS):
how much of an utterance does a trimmed segment cut off at its start/end? Segment edges next to an utterance only."""
import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from groups import *
vs = sys.argv[1].split(","); dets = sys.argv[2].split(",") if len(sys.argv) > 2 else list(DET)
nm = [n for n in names("sinr", None, W_ + P_ + ["clean"])]
for det in dets:
    print("==", det)
    for v in vs:
        cs = []; ce = []
        for n in nm:
            m = MAN[n]; s0 = segs_for(det, n, "T0"); sb = segs_for(det, n, v)
            for (a0, b0), (a, b) in zip(s0, sb):
                if not any(us < a0 < ue for us, ue in m["spans"]):          # the start cut lies in a gap
                    nxt = [us for us, ue in m["spans"] if us >= a0 - 1e-6 and us < b0]
                    if nxt: cs.append(max(0.0, a - nxt[0]))
                if not any(us < b0 < ue for us, ue in m["spans"]):          # the end cut lies in a gap
                    prv = [ue for us, ue in m["spans"] if ue <= b0 + 1e-6 and ue > a0]
                    if prv: ce.append(max(0.0, prv[-1] - b))
        cs = np.array(cs); ce = np.array(ce)
        print(f"  {v:8s} segment starts: cut into the first utterance {100*np.mean(cs>0.0):4.1f}% (>=50ms {100*np.mean(cs>=0.05):4.1f}%, >=100ms {100*np.mean(cs>=0.1):4.1f}%, >=300ms {100*np.mean(cs>=0.3):4.1f}%, max {cs.max()*1000:.0f}ms) | ends: cut into the last utterance {100*np.mean(ce>0.0):4.1f}% (>=100ms {100*np.mean(ce>=0.1):4.1f}%, >=300ms {100*np.mean(ce>=0.3):4.1f}%, max {ce.max()*1000:.0f}ms)   n={len(cs)}")
