#!/usr/bin/env python3
"""Timing of Silero VAD on a 300 s clip, one pinned core, serialized with a lock.

usage: speed_silero.py CORE ROUNDS WAV300 MODEL_DIR PARAKEET_CLI WVAD WHISPER_BIN_DIR OUT.json [--lock FILE]

Jobs (round-robin order, one run each per round):
  wcpp_inproc_detect_only          wvad time: whisper_vad_detect_speech only, model already loaded
  wcpp_vad-speech-segments_cli     whisper-vad-speech-segments (model load + detect + print)
  parakeet_cli_f32 / _f16          parakeet-cli vad (model load + detect + print)
  onnxruntime_python_1thr_detect_only   onnx_time.py (Python loop, one 512-sample call per chunk)
  whisper-cli_tiny.en_novad / _vad      whisper-cli end to end with tiny.en, 1 thread (needs ggml-tiny.en.bin)
Before each run the script waits until the mean number of other runnable tasks over 10 s is below
4 or the 1-minute load average is below 5 (it gives up waiting after 30 tries and runs anyway, and
records quiet=false). It reports wall time, user time, max RSS (from /usr/bin/time -v), the mean
runnable count and the load average at the start. MODEL_DIR: see silero_collect.py.
"""
import fcntl, json, os, re, subprocess, sys, tempfile, time
import numpy as np

CORE, ROUNDS, WAV, M, PK, WV, WB, OUT = sys.argv[1], int(sys.argv[2]), *sys.argv[3:9]
LOCK = sys.argv[sys.argv.index("--lock") + 1] if "--lock" in sys.argv else "/tmp/pk-bench.lock"
WS, WC = WB + "/whisper-vad-speech-segments", WB + "/whisper-cli"
VM = M + "/ggml-silero-v6.2.3-ggml.bin"; TM = M + "/ggml-tiny.en.bin"
HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = {
    "wcpp_inproc_detect_only": [WV, "time", VM, "1", "1", WAV],
    "wcpp_vad-speech-segments_cli": [WS, "-vm", VM, "-f", WAV, "-t", "1", "-np"],
    "parakeet_cli_f32": [PK, "vad", "--model", M + "/silero-vad-f32.gguf", "--input", WAV, "--threads", "1"],
    "parakeet_cli_f16": [PK, "vad", "--model", M + "/silero-vad-f16.gguf", "--input", WAV, "--threads", "1"],
    "onnxruntime_python_1thr_detect_only": ["python3", HERE + "/onnx_time.py", M + "/silero_vad.onnx", WAV],
}
if os.path.exists(TM):
    JOBS["whisper-cli_tiny.en_novad"] = [WC, "-m", TM, "-f", WAV, "-t", "1", "-np", "-nt"]
    JOBS["whisper-cli_tiny.en_vad"] = [WC, "-m", TM, "-f", WAV, "-t", "1", "-np", "-nt", "--vad", "-vm", VM]
INPROC = ("wcpp_inproc", "onnxruntime")

def runnable():
    return int(open("/proc/loadavg").read().split()[3].split("/")[0]) - 1   # minus this process

def gate():
    s = []
    for _ in range(10):
        s.append(runnable()); time.sleep(1)
    return float(np.mean(s)), float(open("/proc/loadavg").read().split()[0])

lockf = open(LOCK, "a")
tf = tempfile.NamedTemporaryFile(suffix=".time", delete=False).name

def timed(name):
    cmd = ["taskset", "-c", CORE, "/usr/bin/time", "-v", "-o", tf] + JOBS[name]
    for attempt in range(30):
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            mr, l1 = gate(); quiet = (mr < 4 or l1 < 5)
            if quiet or attempt == 29:
                t0 = time.perf_counter(); p = subprocess.run(cmd, capture_output=True, text=True); wall = time.perf_counter() - t0
                tt = open(tf).read()
                rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", tt).group(1))
                user = float(re.search(r"User time \(seconds\): ([\d.]+)", tt).group(1))
                inproc = float(p.stdout.strip().split()[0]) if name.startswith(INPROC) else None
                return dict(job=name, wall=wall, inproc=inproc, rss_kb=rss, user=user, mean_runnable=mr, load1=l1, quiet=quiet, attempts=attempt + 1)
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)
        time.sleep(20)

names = list(JOBS); rows = []
for r in range(ROUNDS):
    for n in names[r % len(names):] + names[:r % len(names)]:
        x = timed(n); x["round"] = r; rows.append(x); print(json.dumps(x), flush=True)
os.unlink(tf)
json.dump(rows, open(OUT, "w"), indent=1)
