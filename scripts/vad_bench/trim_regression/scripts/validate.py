import os, subprocess, sys, json, tempfile
sys.path.insert(0, os.path.dirname(__file__)); from common import *; from seg import *; import dec
tot = bad = wbad = 0
for det in DET:
    model, sil = DET[det]
    for name in ["talk_GaryFlake-merged", "sn_heldout_pink0_L00s0", "sn_heldout_white5_L03s1", "sn_tune_clean_L02s0", "ins_GaryFlake-merged_pink-5"]:
        wav = f"{W}/data/{name}.wav"
        import soundfile as sf
        total = sf.info(wav).frames / 16000
        p = probs(det, name)
        for trim in (0.3, 0.0):
            with tempfile.TemporaryDirectory(dir=f"{W}/tmp") as td:
                so = f"{td}/o.txt"
                cmd = [CLI, "transcribe", "--model", model, "--input", wav, "--json", "--threads", "6", "--vad", "--vad-trim", str(trim)] + (["--vad-model", sil] if sil else [])
                r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, PK_SEGOUT=so))
                ref = [l.split("\t") for l in open(so).read().split("\n") if l] if os.path.exists(so) else []
                clitext = json.loads(r.stdout)["text"]
            mine = segments(p, total, OPTS[det], base(trim if trim > 0 else None))
            ok = len(ref) == len(mine) and all(abs(float(a[0]) - s) < 2e-4 and abs(float(a[1]) - e) < 2e-4 for a, (s, e) in zip(ref, mine))
            # text from the harness
            cache = dec.run_segments(det, name, mine, threads=6)
            txt = " ".join(" ".join(w[3] for w in cache[dec.key(s, e)]["words"]) for s, e in mine)
            same = txt.split() == clitext.split()
            tot += 1; bad += not ok; wbad += not same
            print(det, name, trim, "segs", len(ref), len(mine), "bounds_match", ok, "text_match", same, flush=True)
print("segment mismatches", bad, "text mismatches", wbad, "of", tot)
