#!/usr/bin/env python3
"""Plain single-pass vs --vad WER on full-length talks (<name>.wav + <name>.txt).

Like eval_vad_longform.py, but keeps every hypothesis, records wall time and peak
RSS (via /usr/bin/time -v) per run, and survives a failed single pass (a crash or
an out-of-memory kill is recorded as FAIL instead of aborting the sweep).
Existing hypothesis files are reused.  Scoring uses asr_metrics.normalize on both
sides (not the Open ASR Leaderboard normalizer).

  python3 scripts/eval_longform_talks.py --model ultra.gguf --dir /tmp/val/ted --save /tmp/val/ted_res/ultra
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from asr_metrics import wer  # noqa: E402


def run(cli, model, wav, extra, threads, timeout):
    t0 = time.time()
    try:
        r = subprocess.run(["/usr/bin/time", "-v", cli, "transcribe", "--model", model, "--input", str(wav),
                            "--decoder", "tdt", "--threads", str(threads), *extra],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, time.time() - t0, 0, "timeout"
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", r.stderr)
    rss = int(m.group(1)) / 1e6 if m else 0.0
    if r.returncode != 0:
        return None, time.time() - t0, rss, f"exit {r.returncode}: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}"
    return r.stdout.strip(), time.time() - t0, rss, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="build/examples/cli/parakeet-cli")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    save = pathlib.Path(a.save)
    save.mkdir(parents=True, exist_ok=True)
    rows = []
    for wav in sorted(pathlib.Path(a.dir).glob("*.wav")):
        ref = wav.with_suffix(".txt").read_text(encoding="utf-8")
        row = {"talk": wav.stem}
        for mode, extra in (("plain", []), ("vad", ["--vad"])):
            hp, mp = save / f"{wav.stem}.{mode}.txt", save / f"{wav.stem}.{mode}.json"
            if mp.exists():
                row[mode] = json.loads(mp.read_text())
                continue
            hyp, secs, rss, err = run(a.cli, a.model, wav, extra, a.threads, a.timeout)
            res = {"wall_s": round(secs, 1), "rss_gb": round(rss, 2), "err": err,
                   "wer": None if hyp is None else round(100 * wer(ref, hyp), 3)}
            if hyp is not None:
                hp.write_text(hyp + "\n", encoding="utf-8")
            mp.write_text(json.dumps(res))
            row[mode] = res
            print(wav.stem, mode, res, flush=True)
        rows.append(row)
    ok = [r for r in rows if r["plain"]["wer"] is not None and r["vad"]["wer"] is not None]
    print("talk\tplain_wer\tvad_wer\tplain_s\tvad_s\tplain_rss_gb\tvad_rss_gb")
    for r in rows:
        p, v = r["plain"], r["vad"]
        print(f"{r['talk']}\t{p['wer'] if p['wer'] is not None else 'FAIL'}\t{v['wer'] if v['wer'] is not None else 'FAIL'}"
              f"\t{p['wall_s']}\t{v['wall_s']}\t{p['rss_gb']}\t{v['rss_gb']}")
    for mode in ("plain", "vad"):
        v = [r[mode]["wer"] for r in rows if r[mode]["wer"] is not None]
        if v:
            print(f"MEAN {mode}: {sum(v) / len(v):.2f} over {len(v)}/{len(rows)} talks")
    if ok:
        print(f"MEAN on the {len(ok)} talks where both ran: plain {sum(r['plain']['wer'] for r in ok) / len(ok):.2f} "
              f"vad {sum(r['vad']['wer'] for r in ok) / len(ok):.2f}")

if __name__ == "__main__":
    main()
