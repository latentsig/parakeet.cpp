#!/usr/bin/env python3
"""Reference sanity: frame energy (dB, 25 ms window / 10 ms hop) in cells that are (a) ref speech and detected by Silero .5, (b) ref speech missed by Silero, (c) ref non-speech and called speech by Redux head only, (d) ref non-speech and not detected. Per domain medians relative to the file's 10th percentile energy."""
import numpy as np, soundfile as sf, vr, systems
D = vr.load_recordings(); rows = {}
for m in D:
    if m["nonspeech"]: continue
    y, _ = sf.read(m["wav"], dtype="float32"); n = len(m["ref"]); fr = np.lib.stride_tricks.sliding_window_view(np.pad(y, (0, 400)), 400)[::160][:n]
    e = 10 * np.log10((fr ** 2).mean(1) + 1e-10); floor = np.percentile(e, 10); e = e - floor
    r = systems.Rec(m); S = systems.mask(r, ("sil", 0.5)); H = systems.mask(r, ("head", "redux", 0.5)); ref = m["ref"]
    for nm, sel in (("ref speech, Silero hit", ref & S), ("ref speech, Silero miss", ref & ~S), ("ref speech, Redux miss", ref & ~H), ("ref non-speech, Redux only", ~ref & H & ~S), ("ref non-speech, neither", ~ref & ~H & ~S)):
        rows.setdefault((m["domain"], nm), []).append((sel.sum(), np.median(e[sel]) if sel.sum() else np.nan))
print("| domain | cells | share of domain cells | median level above file noise floor (dB) |\n|---|---|---|---|")
for d in ("vox", "ami", "ava"):
    tot = sum(m["ref"].size for m in D if m["domain"] == d)
    for nm in ("ref speech, Silero hit", "ref speech, Silero miss", "ref speech, Redux miss", "ref non-speech, Redux only", "ref non-speech, neither"):
        v = rows[(d, nm)]; n = sum(a for a, _ in v); med = np.nanmedian([b for a, b in v if a > 100])
        print(f"| {d} | {nm} | {100*n/tot:.1f}% | {med:.1f} |")
