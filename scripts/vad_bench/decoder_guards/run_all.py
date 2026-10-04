#!/usr/bin/env python3
"""usage: run_all.py DATA OUT --old CLI_OLD --new CLI_NEW --ultra GGUF --redux GGUF --v3 GGUF --silero GGUF [-j N]

Runs the CLI over the corpus of make_corpus.py and keeps every JSON output in OUT. CLI_OLD is
parakeet-cli built from the commit before the trim change, CLI_NEW the one with it. Per detector
(ultra head, redux head, v3 with Silero):
  <det>.old/<file>.json      transcribe --vad --json with CLI_OLD
  <det>.new/<file>.json      the same with CLI_NEW (trim 0.3, the default)
  <det>.new0/<file>.json     CLI_NEW with --vad-trim 0 (must equal .old byte for byte; a subset)
  <det>.f05/<file>.json      CLI_NEW with --min-local-conf 0.5 (ultra and v3 only)
  <det>.plain/<file>.json, <det>.plainf05/<file>.json   noise-only files, no VAD, without and with the filter
  <det>.old0f05/<file>.json  CLI_NEW with --vad-trim 0 --min-local-conf 0.5, only on the insert files in whose noise
                             block the old cuts gave words (the filter on its own, without the trim)
  <det>.f07, <det>.f09/<file>.json  thresholds 0.7 and 0.9 on the speech in noise files (ultra and v3)
  <det>.segold/<file>.json, <det>.segnew/<file>.json   `vad --mode segments` of insert files (cuts)
Existing outputs are kept, so a run can be resumed. Not a timing run."""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("data"); ap.add_argument("out")
for k in ("old", "new", "ultra", "redux", "v3", "silero"): ap.add_argument("--" + k, required=True)
ap.add_argument("--only", default="", help="regular expression: run only the files whose name matches")
ap.add_argument("-j", type=int, default=3); ap.add_argument("--sets", type=int, default=4, help="LibriSpeech sets per condition")
a = ap.parse_args()
man = [m for m in json.load(open(f"{a.data}/manifest.json"))
       if m["kind"] != "sinr" or int(m["name"].rsplit("_", 1)[1]) < a.sets]
import re
if a.only: man = [m for m in man if re.search(a.only, m["name"])]
DET = {"ultra": (a.ultra, None), "redux": (a.redux, None), "v3": (a.v3, a.silero)}
jobs = []
def add(out, cmd):
    if not os.path.exists(out) or os.path.getsize(out) == 0: jobs.append((out, cmd))
for det, (model, sil) in DET.items():
    vad = ["--vad"] + (["--vad-model", sil] if sil else [])
    for mf in man:
        wav = f"{a.data}/{mf['name']}.wav"
        if mf["kind"] == "noise":   # no VAD: the whole file is decoded, every word is an invented word
            plain = ["transcribe", "--model", model, "--input", wav, "--json"]
            for tag, extra in (("plain", []), ("plainf05", ["--min-local-conf", "0.5"])):
                if tag == "plainf05" and det == "redux": continue
                os.makedirs(f"{a.out}/{det}.{tag}", exist_ok=True)
                add(f"{a.out}/{det}.{tag}/{mf['name']}.json", [a.new] + plain + extra)
            continue
        base = ["transcribe", "--model", model, "--input", wav, "--json"] + vad
        def o(tag): os.makedirs(f"{a.out}/{det}.{tag}", exist_ok=True); return f"{a.out}/{det}.{tag}/{mf['name']}.json"
        add(o("old"), [a.old] + base)
        add(o("new"), [a.new] + base)
        if mf["kind"] == "talk" or mf["name"].endswith(("_00", "_01")) or mf["name"].endswith(("white-20", "music-35")):
            add(o("new0"), [a.new] + base + ["--vad-trim", "0"])
        if det != "redux":
            add(o("f05"), [a.new] + base + ["--min-local-conf", "0.5"])
        if mf["kind"] == "insert":
            seg = ["vad", "--model", sil or model, "--input", wav, "--mode", "segments"]
            add(o("segold"), [a.old] + seg)
            add(o("segnew"), [a.new] + seg)
print(len(jobs), "jobs", flush=True)
def run(j):
    out, cmd = j
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0: open(out, "w").write(r.stdout)
    else: print("FAILED", out, r.stderr[-300:], flush=True)
    return out
with ThreadPoolExecutor(a.j) as ex:
    for i, o in enumerate(ex.map(run, jobs)):
        if i % 20 == 0: print(i, "/", len(jobs), flush=True)

# Second pass: the filter alone, on the files where the old cuts gave invented words in the noise block.
jobs = []
for det, (model, sil) in DET.items():
    vad = ["--vad"] + (["--vad-model", sil] if sil else [])
    if det == "redux": continue
    for mf in man:
        if mf["kind"] != "insert": continue
        f = f"{a.out}/{det}.old/{mf['name']}.json"
        if not os.path.exists(f) or os.path.getsize(f) == 0: continue
        lo, hi = mf["ins"]
        if not any(lo + 0.5 <= w["start"] <= hi - 0.5 for w in json.load(open(f))["words"]): continue
        os.makedirs(f"{a.out}/{det}.old0f05", exist_ok=True)
        add(f"{a.out}/{det}.old0f05/{mf['name']}.json",
            [a.new, "transcribe", "--model", model, "--input", f"{a.data}/{mf['name']}.wav", "--json"] + vad + ["--vad-trim", "0", "--min-local-conf", "0.5"])
for det, (model, sil) in DET.items():
    if det == "redux": continue
    vad = ["--vad"] + (["--vad-model", sil] if sil else [])
    for mf in man:
        if mf["kind"] != "sinr": continue
        for tag, thr in (("f07", "0.7"), ("f09", "0.9")):
            os.makedirs(f"{a.out}/{det}.{tag}", exist_ok=True)
            add(f"{a.out}/{det}.{tag}/{mf['name']}.json",
                [a.new, "transcribe", "--model", model, "--input", f"{a.data}/{mf['name']}.wav", "--json"] + vad + ["--min-local-conf", thr])
print(len(jobs), "filter-only jobs", flush=True)
with ThreadPoolExecutor(a.j) as ex:
    list(ex.map(run, jobs))
