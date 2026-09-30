#!/usr/bin/env python3
"""WER of plain single-pass vs --vad transcription on long clips with references.

Use --skip-plain to reuse a plain-pass result when sweeping VAD parameters.
"""
import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from asr_metrics import wer  # noqa: E402


def run(cmd_prefix, cli, model, wav, extra, threads):
    try:
        r = subprocess.run([*cmd_prefix, cli, "transcribe", "--model", model, "--input", wav,
                            "--decoder", "tdt", "--threads", str(threads), *extra],
                           capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"parakeet-cli failed (exit {e.returncode}) on {wav}:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="build/examples/cli/parakeet-cli")
    ap.add_argument("--model", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--glob", default="*.wav")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prefix", default="", help="command prefix, e.g. 'taskset -c 0-7'")
    ap.add_argument("--skip-plain", action="store_true")
    ap.add_argument("--save", help="directory to keep the raw transcripts")
    ap.add_argument("--vad-arg", action="append", default=[],
                    help="extra args for the --vad run, e.g. --vad-arg=--vad-threshold=0.4 (repeatable)")
    args = ap.parse_args()
    extra = [a for v in args.vad_arg for a in v.split("=", 1)]
    prefix = args.prefix.split()
    save = pathlib.Path(args.save) if args.save else None
    if save:
        save.mkdir(parents=True, exist_ok=True)
    tot_plain = tot_vad = n = 0.0
    wavs = sorted(pathlib.Path(args.dir).glob(args.glob))
    if not wavs:
        print(f"no files match {args.glob} in {args.dir}", file=sys.stderr)
        sys.exit(2)
    for wav in wavs:
        ref = wav.with_suffix(".txt").read_text()
        wp = float("nan")
        if not args.skip_plain:
            plain = run(prefix, args.cli, args.model, str(wav), [], args.threads)
            wp = wer(ref, plain)
            if save:
                (save / (wav.stem + ".plain.txt")).write_text(plain + "\n")
        vad = run(prefix, args.cli, args.model, str(wav), ["--vad", *extra], args.threads)
        wv = wer(ref, vad)
        if save:
            (save / (wav.stem + ".vad.txt")).write_text(vad + "\n")
        print(f"{wav.name:32s} plain {wp*100:5.2f}%   vad {wv*100:5.2f}%")
        tot_plain += wp; tot_vad += wv; n += 1
    print(f"{'mean':32s} plain {tot_plain/n*100:5.2f}%   vad {tot_vad/n*100:5.2f}%")


if __name__ == "__main__":
    main()
