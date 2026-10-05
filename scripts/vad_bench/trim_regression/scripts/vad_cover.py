"""How far outside the VAD speech mask are the (correctly) decoded words of the untrimmed run?
For each correct word of the T0 decode: gap = distance from the word interval to the nearest speech frame of the smoothed mask
(0 = overlaps speech; >0 = the word lies outside the mask, before ('lead') or after ('lag') the speech run)."""
import sys, collections; sys.path.insert(0, __import__('os').path.dirname(__file__))
from groups import *
dets = sys.argv[1].split(",") if len(sys.argv) > 1 else list(DET)
for det in dets:
    o = OPTS[det]; fs = o["frame_sec"]
    for kind, nm in (("talks", names("talk")), ("sinr clean", names("sinr", None, ["clean"])), ("sinr noisy", names("sinr", None, W_ + P_)), ("sinr pink0+white0", names("sinr", None, ["pink0", "white0"]))):
        lead = []; lag = []; tot = 0
        for n in nm:
            try: segs = segs_for(det, n, "T0"); hy = hyp(det, n, segs); ev = align(norm(MAN[n]["ref"]), hy)
            except KeyError: continue
            st = {hj: k for k, rj, hj in ev if hj is not None}
            tt = total_sec(n); nf = int(np.ceil(tt / fs - 1e-9)); sp, _ = mask(probs(det, n), nf, o)
            idx = np.where(sp)[0]; starts = idx * fs; ends = (idx + 1) * fs
            for hj, (tok, s, e, si, cf) in enumerate(hy):
                if st[hj] != "C": continue
                tot += 1
                # nearest speech frame
                k = np.searchsorted(starts, s)
                # speech overlapping the word?
                ov = np.any((starts < e) & (ends > s)) if len(idx) else False
                if ov: lead.append(0.0); lag.append(0.0); continue
                prev_end = ends[ends <= s].max() if np.any(ends <= s) else -9
                next_start = starts[starts >= e].min() if np.any(starts >= e) else 1e9
                lead.append(max(0.0, next_start - e) if next_start - e < s - prev_end else 0.0)   # word is before the next speech run
                lag.append(max(0.0, s - prev_end) if s - prev_end <= next_start - e else 0.0)    # word is after the previous speech run
        lead = np.array(lead); lag = np.array(lag)
        print(f"{det:6s}{kind:18s} correct words {tot:6d} | outside the mask (any) {100*np.mean((lead>0)|(lag>0)):4.2f}%  start-side>0.1s {100*np.mean(lead>0.1):4.2f}%  >0.3s {100*np.mean(lead>0.3):4.2f}%  end-side>0.1s {100*np.mean(lag>0.1):4.2f}% >0.3s {100*np.mean(lag>0.3):4.2f}%")
