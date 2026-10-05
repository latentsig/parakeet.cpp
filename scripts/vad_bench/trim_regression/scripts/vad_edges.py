"""Onset/offset timing of the VAD against the true utterance spans (sinr files). No ASR."""
import sys, os, numpy as np, collections
sys.path.insert(0, os.path.dirname(__file__)); from lib import *
files = [m for m in MAN.values() if m["kind"] == "sinr" and m["dur"] > 30]
def cond_family(c): return "clean" if c == "clean" else c
rows = collections.defaultdict(list)   # (det, family, kind) -> list of values
prof = collections.defaultdict(lambda: collections.defaultdict(list))  # (det, fam, 'on'|'off') -> rel offset -> list of speech flags
REL = np.round(np.arange(-0.48, 0.97, 0.08), 2)
for det in DET:
    o = OPTS[det]; fs = o["frame_sec"]
    for m in files:
        p = probs(det, m["name"]); total = m["dur"]; n = int(np.ceil(total / fs - 1e-9))
        sp, _ = mask(p, n, o); raw = np.zeros(n, bool); raw[:min(n, len(p))] = p[:n] >= o["threshold"]
        segs = segments(p, total, o, None); cuts = [s for s, e in segs[1:]] + [e for s, e in segs[:-1]]
        for ui, (us, ue) in enumerate(m["spans"]):
            f0 = max(0, int((us - 0.5) / fs)); f1 = min(n, int(np.ceil(ue / fs)))
            fr = [f for f in range(f0, f1) if sp[f] and (f + 1) * fs > us - 0.5]
            first = next((f for f in range(f0, f1) if sp[f]), None)
            g0 = max(0, int(us / fs)); g1 = min(n, int(np.ceil((ue + 0.5) / fs)))
            last = next((f for f in range(g1 - 1, g0 - 1, -1) if sp[f]), None)
            # edge utterance: next to a cut (cut within 1.5 s before start / after end)
            edge_on = any(us - 3.0 <= c <= us for c in cuts); edge_off = any(ue <= c <= ue + 3.0 for c in cuts)
            fam = cond_family(m["cond"])
            for tag, ok in (("all", True), ("edge", None)):
                pass
            late = None if first is None else first * fs - us
            early = None if last is None else ue - (last + 1) * fs
            for fam_ in (fam, "ALL-noisy" if fam != "clean" else "clean", "ALL"):
                rows[(det, fam_, "on_all")].append(late); rows[(det, fam_, "off_all")].append(early)
                if edge_on: rows[(det, fam_, "on_edge")].append(late)
                if edge_off: rows[(det, fam_, "off_edge")].append(early)
            for r in REL:
                fo = int(np.floor((us + r) / fs)); fe = int(np.floor((ue + r - 0.0) / fs))
                if 0 <= fo < n: prof[(det, "ALL-noisy" if fam != "clean" else "clean", "on")][r].append(raw[fo])
                if 0 <= fe < n: prof[(det, "ALL-noisy" if fam != "clean" else "clean", "off")][r].append(raw[fe])
def stats(v):
    miss = sum(x is None for x in v); x = np.array([a for a in v if a is not None])
    return f"n={len(v):4d} miss={100*miss/len(v):4.1f}% median={np.median(x)*1000:5.0f}ms p90={np.percentile(x,90)*1000:5.0f} p95={np.percentile(x,95)*1000:5.0f} >0.3s={100*(x>0.3).mean():4.1f}% >0.5s={100*(x>0.5).mean():4.1f}%"
print("Lateness of the speech start (VAD smoothed mask start minus true utterance start) and earliness of the end (true end minus VAD end). Positive = the VAD cuts into the word.")
for fam in ("clean", "white20", "white10", "white5", "white0", "pink20", "pink10", "pink5", "pink0", "ALL-noisy"):
    print(f"-- {fam}")
    for det in DET:
        for k in ("on_all", "on_edge", "off_all", "off_edge"):
            if (det, fam, k) in rows: print(f"  {det:6s}{k:8s}", stats(rows[(det, fam, k)]))
print("\nFraction of true onsets (offsets) at which the RAW frame (p>=0.5) at time t relative to the true onset (end) is speech. t in s.")
for fam in ("clean", "ALL-noisy"):
    for kind in ("on", "off"):
        print(f"-- {fam} {kind}set   t=" + " ".join(f"{r:+.2f}" for r in REL[::2]))
        for det in DET:
            print(f"  {det:6s}      " + " ".join(f"{100*np.mean(prof[(det, fam, kind)][r]):5.0f}" for r in REL[::2]))
