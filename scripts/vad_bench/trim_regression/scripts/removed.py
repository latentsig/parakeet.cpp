import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from groups import *
for det in DET:
    for kind, nm in (("talk", names("talk")), ("sinr-noisy", names("sinr", None, W_ + P_)), ("sinr-clean", names("sinr", None, ["clean"]))):
        rem = []; nseg = 0; tot = 0.0
        for n in nm:
            s0 = segs_for(det, n, "T0"); s3 = segs_for(det, n, "T0.3")
            assert len(s0) == len(s3)
            for (a, b), (c, d) in zip(s0, s3):
                nseg += 1; tot += b - a
                for x in (c - a, b - d):
                    rem.append(x)
        rem = np.array(rem)
        # exclude file start/end edges? keep all
        print(f"{det:6s}{kind:11s} segments {nseg:5d} mean len {tot/nseg:5.1f}s | edges removed: >0 {100*(rem>1e-6).mean():4.1f}%  >=0.1s {100*(rem>=0.1).mean():4.1f}%  >=0.5s {100*(rem>=0.5).mean():4.1f}%  >=1s {100*(rem>=1).mean():4.1f}%  >=2s {100*(rem>=2).mean():4.1f}%  mean {rem.mean():.2f}s  p90 {np.percentile(rem,90):.2f}s")
