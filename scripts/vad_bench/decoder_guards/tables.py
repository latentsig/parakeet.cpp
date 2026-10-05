#!/usr/bin/env python3
"""usage: tables.py DATA OUT > results/tables.txt

Reads the outputs of run_all.py and prints the tables of docs/vad-benchmarks.md (trim and word filter)."""
import json, os, re, sys, glob
import numpy as np

D, O = sys.argv[1], sys.argv[2]
man = {m["name"]: m for m in json.load(open(f"{D}/manifest.json"))}
DETS = [("ultra", "Ultra head"), ("redux", "Redux head"), ("v3", "v3 + Silero")]

def norm(s): return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ")).split()
def ed(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1): cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]
def load(det, tag, name):
    f = f"{O}/{det}.{tag}/{name}.json"
    if not os.path.exists(f) or os.path.getsize(f) == 0: return None
    return json.load(open(f))
def wer(det, tag, names):
    e = n = 0
    for nm in names:
        j = load(det, tag, nm)
        if j is None: return None
        ref = norm(man[nm]["ref"]); e += ed(norm(j["text"]), ref); n += len(ref)
    return 100.0 * e / n, n
def fmt(x): return "-" if x is None else f"{x[0]:.2f}"
def overlap(segs, a, b): return sum(max(0.0, min(s["end"], b) - max(s["start"], a)) for s in segs)

print("uptime when the tables were made:", open("/proc/loadavg").read().split()[:3], "(load average 1, 5, 15 min)")
print("\n== byte check: --vad-trim 0 against the old binary (JSON output, every file run both ways)")
for det, lab in DETS:
    eq = tot = 0
    for f in glob.glob(f"{O}/{det}.new0/*.json"):
        nm = os.path.basename(f)[:-5]; o = f"{O}/{det}.old/{nm}.json"
        if os.path.exists(o): tot += 1; eq += open(f).read() == open(o).read()
    print(f"{lab:12s} {eq}/{tot} identical")

print("\n== word error rate (%), words in the reference in brackets")
groups = [("talks", [n for n in man if man[n]["kind"] == "talk"])]
for c in ("clean", "white5", "pink0"):
    groups.append((f"speech in noise: {c}", [n for n in man if man[n]["kind"] == "sinr" and man[n]["cond"] == c and load("ultra", "old", n) is not None]))
for gname, names in groups:
    nref = sum(len(norm(man[n]["ref"])) for n in names)
    print(f"-- {gname}: {len(names)} files, {nref} words")
    print(f"{'':14s}{'old':>8s}{'trim 0.3':>10s}{'+filter 0.5':>13s}{'trim - old':>12s}{'filter - trim':>15s}")
    for det, lab in DETS:
        o, t, f = wer(det, "old", names), wer(det, "new", names), wer(det, "f05", names)
        d1 = "-" if not (o and t) else f"{t[0] - o[0]:+.2f}"
        d2 = "-" if not (t and f) else f"{f[0] - t[0]:+.2f}"
        print(f"{lab:14s}{fmt(o):>8s}{fmt(t):>10s}{fmt(f):>13s}{d1:>12s}{d2:>15s}")
for tk in sorted({m["talk"] for m in man.values() if m["kind"] == "talk"}):
    print(f"-- talk {tk}")
    for det, lab in DETS:
        o, t, f = wer(det, "old", [f"talk_{tk}"]), wer(det, "new", [f"talk_{tk}"]), wer(det, "f05", [f"talk_{tk}"])
        print(f"{lab:14s}{fmt(o):>8s}{fmt(t):>10s}{fmt(f):>13s}")

print("\n== noise block of 60 s inside speech (insert files): seconds of the block that the decoder gets, and words in it")
ins = [n for n in man if man[n]["kind"] == "insert"]
print(f"files: {len(ins)} (3 talks x white, pink, clicks, music at -20 and -5 dB, white and music at -35 dB)")
print(f"{'':14s}{'sec old':>9s}{'sec new':>9s}{'words old':>11s}{'words new':>11s}{'+filter 0.5':>13s}{'files with words old/new/f05':>32s}")
def inblock(j, a, b):
    return [w for w in j["words"] if a + 0.5 <= w["start"] <= b - 0.5]
for det, lab in DETS:
    so, sn, wo, wn, wf, fo, fn, ff = [], [], 0, 0, 0, 0, 0, 0
    for nm in ins:
        a, b = man[nm]["ins"]
        jo, jn, jf = load(det, "old", nm), load(det, "new", nm), load(det, "f05", nm)
        so_, sn_ = load(det, "segold", nm), load(det, "segnew", nm)
        if so_ and sn_: so.append(overlap(so_["segments"], a, b)); sn.append(overlap(sn_["segments"], a, b))
        if jo: k = len(inblock(jo, a, b)); wo += k; fo += k > 0
        if jn: k = len(inblock(jn, a, b)); wn += k; fn += k > 0
        if jf: k = len(inblock(jf, a, b)); wf += k; ff += k > 0
    print(f"{lab:14s}{np.mean(so):9.1f}{np.mean(sn):9.1f}{wo:11d}{wn:11d}{(str(wf) if det != 'redux' else '-'):>13s}{f'{fo}/{fn}/{ff}' if det != 'redux' else f'{fo}/{fn}/-':>32s}")
print("per noise type, seconds of the block decoded (old -> new):")
for det, lab in DETS:
    row = {}
    for nm in ins:
        a, b = man[nm]["ins"]; so_, sn_ = load(det, "segold", nm), load(det, "segnew", nm)
        if so_ and sn_: row.setdefault(f"{man[nm]['noise']}{man[nm]['level']}", []).append((overlap(so_["segments"], a, b), overlap(sn_["segments"], a, b)))
    print(f"{lab:14s}" + "  ".join(f"{k} {np.mean([x[0] for x in v]):.0f}->{np.mean([x[1] for x in v]):.0f}" for k, v in sorted(row.items())))

print("\n== what the filter (0.5) took out of the files with real speech")
for det, lab in DETS:
    if det == "redux": continue
    dropped = 0; words = 0; hall = 0
    for nm, m in man.items():
        j, t = load(det, "f05", nm), load(det, "new", nm)
        if j is None or t is None: continue
        d = j.get("guard", {}).get("dropped_words", 0)
        if m["kind"] == "insert":
            a, b = m["ins"]; hall += len(inblock(t, a, b)) - len(inblock(j, a, b))
            # words outside the block that went with it
            dropped += d - (len(inblock(t, a, b)) - len(inblock(j, a, b)))
        else: dropped += d
        words += len(t["words"])
    print(f"{lab:12s} words decoded {words}, hallucinated words in the noise blocks removed {hall}, other words removed {dropped}")

print("\n== 30 s of noise alone, decoded whole (no VAD): every word is an invented word")
nz = [n for n in man if man[n]["kind"] == "noise"]
print(f"files: {len(nz)} (3 talks x white, pink, clicks, music, hum, tone, sweep at -35, -20, -5 dB against the speech level)")
print(f"{'':14s}{'files with words':>18s}{'words':>8s}{'+filter 0.5: files':>20s}{'words':>8s}")
for det, lab in DETS:
    fw = w = ffw = fw5 = 0
    for nm in nz:
        j, f = load(det, "plain", nm), load(det, "plainf05", nm)
        if j is None: continue
        k = len(j["words"]); w += k; fw += k > 0
        if f is not None: fw5 += len(f["words"]); ffw += len(f["words"]) > 0
    print(f"{lab:14s}{fw:18d}{w:8d}{(str(ffw) if det != 'redux' else '-'):>20s}{(str(fw5) if det != 'redux' else '-'):>8s}")

print("\n== the filter alone (old cuts, --vad-trim 0, --min-local-conf 0.5) on the insert files whose noise block gave words")
for det, lab in DETS:
    if det == "redux": continue
    files = hall = left = lost = real = 0
    for nm in ins:
        j, f = load(det, "old", nm), load(det, "old0f05", nm)
        if j is None or f is None: continue
        a, b = man[nm]["ins"]; files += 1
        k0, k1 = len(inblock(j, a, b)), len(inblock(f, a, b))
        hall += k0; left += k1
        lost += len(j["words"]) - len(f["words"]) - (k0 - k1); real += len(j["words"]) - k0
    print(f"{lab:12s} {files} files: words in the block {hall} -> {left}; other words of those files {real}, lost {lost}")

print("\n== higher thresholds on the speech in noise files (WER %, trim 0.3; words dropped of the words decoded)")
sn = [n for n in man if man[n]["kind"] == "sinr" and load("ultra", "f05", n) is not None]
for det, lab in DETS:
    if det == "redux": continue
    row = []
    for tag, t in (("new", "off"), ("f05", "0.5"), ("f07", "0.7"), ("f09", "0.9")):
        w = wer(det, tag, [n for n in sn if load(det, tag, n) is not None])
        dr = sum(load(det, tag, n).get("guard", {}).get("dropped_words", 0) for n in sn if load(det, tag, n) is not None)
        row.append(f"{t}: {fmt(w)} ({dr} dropped)")
    print(f"{lab:12s} {len(sn)} files, {sum(len(norm(man[n]['ref'])) for n in sn)} reference words   " + "   ".join(row))
