#!/usr/bin/env python3
"""WER of transcribe --vad style decoding per segmentation system, paired block bootstrap (100 reference words per block, resampled over all talks)."""
import json, os, re, sys, numpy as np, jiwer
ROOT = os.environ.get("VAD_REAL_ROOT") or os.getcwd()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr
MODE = sys.argv[1] if len(sys.argv) > 1 else "ted"
plan = json.load(open(f"{ROOT}/results/wer_plan.json" if MODE == "ted" else f"{ROOT}/results/wer_plan_{MODE}.json")); SET = plan.pop("_set", "ted")
CD = "wercache" if SET == "ted" else f"wercache_{SET}"
REFD = "ted" if MODE in ("ted", "forced") else "comp"
def norm(s): return re.sub(r"[^a-z0-9' ]", " ", s.lower().replace("-", " ")).split()
def key(s, e): return f"{s:.4f},{e:.4f}"
def hyp(asr, talk, segs):
    cache = json.load(open(f"{ROOT}/results/{CD}/{asr}/{talk}.json")); toks = []
    for s, e in segs:
        for w in cache[key(s, e)]["words"]: toks += norm(w[3])
    return toks
BL = 100
def errs(ref, h):
    """per-block [errors, ref words], plus S D I"""
    nb = (len(ref) + BL - 1) // BL; blk = np.zeros((nb, 2), np.int64); sdi = np.zeros(3, np.int64)
    for i in range(len(ref)): blk[i // BL, 1] += 1
    if not h: blk[:, 0] = blk[:, 1]; sdi[1] = len(ref); return blk, sdi
    r = jiwer.process_words(" ".join(ref), " ".join(h)); last = 0
    for ch in r.alignments[0]:
        if ch.type == "equal": last = ch.ref_end_idx - 1; continue
        if ch.type in ("substitute", "delete"):
            for k in range(ch.ref_start_idx, ch.ref_end_idx):
                blk[min(k, len(ref) - 1) // BL, 0] += 1; last = k
            sdi[0 if ch.type == "substitute" else 1] += ch.ref_end_idx - ch.ref_start_idx
        else:
            n = ch.hyp_end_idx - ch.hyp_start_idx; blk[min(max(ch.ref_start_idx - 1, 0), len(ref) - 1) // BL, 0] += n; sdi[2] += n
    return blk, sdi
out = [f"# WER ({MODE}), decoding the segments of each system", ""]
SYS = ["head", "silero", "fusion_def", "fusion_tuned", "gate92", "orgate92"] if MODE != "forced" else ["pause20", "forced20"]
rng = np.random.default_rng(5)
for asr in ("ultra", "redux"):
    talks = list(plan[asr].keys())
    if not all(os.path.exists(f"{ROOT}/results/{CD}/{asr}/{t}.json") for t in talks): continue
    R = {}; blocks = {}; sdis = {}; dec = {}
    for sname in SYS:
        allb = []; sd = np.zeros(3, np.int64); secs = 0.0; per = []
        for t in talks:
            ref = norm(open(f"{ROOT}/data/{REFD}/{t}.txt").read()); segs = plan[asr][t][sname]
            b, s = errs(ref, hyp(asr, t, segs)); allb.append(b); sd += s; secs += sum(e - a for a, e in segs); per.append(b[:, 0].sum() / b[:, 1].sum())
        blocks[sname] = np.concatenate(allb); sdis[sname] = sd; dec[sname] = (secs, per)
    B0 = SYS[0]; B1 = "silero" if MODE != "forced" else "forced20"
    nb = len(blocks[B0]); bi = rng.integers(0, nb, size=(3000, nb))
    def wer(b, idx=None):
        x = b if idx is None else b[idx]; return x[..., 0].sum(-1) / x[..., 1].sum(-1)
    out += [f"## {MODE}: {asr} ASR model (VAD segmentation from {asr} head / Silero / fusion / gate)", "", f"| system | WER % [95% CI] | S | D | I | decoded s | per-talk WER % | dWER vs {B0} (pp) | dWER vs {B1} (pp) |", "|---|---|---|---|---|---|---|---|---|"]
    for sname in SYS:
        b = blocks[sname]; w = wer(b); ws = wer(b, bi); l, h = np.percentile(ws, [2.5, 97.5])
        d1 = ws - wer(blocks[B0], bi); d2 = ws - wer(blocks[B1], bi)
        c1 = f"{100*(w-wer(blocks[B0])):+.2f} [{100*np.percentile(d1,2.5):+.2f}, {100*np.percentile(d1,97.5):+.2f}]"
        c2 = f"{100*(w-wer(blocks[B1])):+.2f} [{100*np.percentile(d2,2.5):+.2f}, {100*np.percentile(d2,97.5):+.2f}]"
        out.append(f"| {sname} | {100*w:.2f} [{100*l:.2f}, {100*h:.2f}] | {sdis[sname][0]} | {sdis[sname][1]} | {sdis[sname][2]} | {dec[sname][0]:.0f} | " + ", ".join(f"{100*x:.1f}" for x in dec[sname][1]) + f" | {c1} | {c2} |")
    out.append("")
out.append("talk order: " + ", ".join(plan['ultra'].keys()))
open(f"{ROOT}/results/wer.md" if MODE == "ted" else f"{ROOT}/results/wer_{MODE}.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
