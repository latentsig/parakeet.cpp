# Trim and word filter runs

Scripts behind the section "Trimming segments and the word filter" of
[docs/vad-benchmarks.md](../../../docs/vad-benchmarks.md). No audio and no model files are
committed. `results/` holds the output of the run on that page.

`CLI_OLD` is the build of master at the commit before this change (91b120b), `CLI_NEW` the build of
the change. Python packages: `numpy soundfile librosa datasets`. Models: the Ultra and Redux GGUFs
(`moondream/parakeet-ultra`, `moondream/parakeet-redux`, F16 and dequantized), a TDT 0.6B v3 F16
GGUF, and a Silero GGUF (`scripts/convert_silero_vad_to_gguf.py`).

```
python3 fetch_data.py DATA                  # 3 TED-LIUM talks (not the first four) and LibriSpeech test-clean
python3 make_corpus.py DATA                 # inserts of synthetic noise, speech in noise, manifest.json
python3 run_all.py DATA OUT --old CLI_OLD --new CLI_NEW --ultra ultra-f16.gguf \
    --redux redux-deq-f16.gguf --v3 tdt-0.6b-v3-f16.gguf --silero silero-vad-f16.gguf -j 3
python3 tables.py DATA OUT > results/tables.txt
rm -r DATA                                  # audio
```

`fetch_data.py` streams the data; nothing else is downloaded. `make_corpus.py` uses fixed seeds,
so the files are the same on every run. `run_all.py` needs a few hours of CPU on a loaded machine
and resumes when you start it again. It is not a timing run: the machine was loaded (the load
average is in `results/tables.txt`).

What the files are:

- `talk_*`: three whole talks, with the reference text of the dataset.
- `ins_<talk>_<noise><level>`: 90 s of a talk with 60 s of synthetic noise (white, pink, clicks,
  music-like tones) inserted at a quiet point, at -20 or -35 dB against the speech. The noise block is
  where every word is an invented word.
- `sn_<cond>_<k>`: six LibriSpeech utterances with gaps, clean, with white noise at 5 dB SNR or
  with pink noise at 0 dB SNR.

`tables.py` prints, for each detector (Ultra head, Redux head, v3 with Silero): the check that
`--vad-trim 0` gives the old output byte for byte, the WER with the old cuts, with trim 0.3 and with
trim 0.3 plus `--min-local-conf 0.5`, the seconds of the noise block that the decoder gets, and the
words it returns inside the block.
