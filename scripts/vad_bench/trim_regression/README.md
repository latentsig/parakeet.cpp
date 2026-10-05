# Trim regression follow-up

Scripts and result files behind the section "Trimming: word error rate, enlarged measurement" of
[docs/vad-benchmarks.md](../../../docs/vad-benchmarks.md). The first run of the trim change (see
[../decoder_guards](../decoder_guards/README.md)) read a small loss for the Redux head. This
follow-up repeated the comparison on 9 talks and about 375 noisy files with paired bootstrap
intervals and found that the loss was noise.

No audio, no model files, no probability files and no decode cache are committed. The tables that
the page quotes are in [../decoder_guards/results/trim_regression_followup.md](../decoder_guards/results/trim_regression_followup.md).

## What is here

| Path | What |
| --- | --- |
| `scripts/` | Python scripts. `common.py` holds the paths and the segmenter settings, `seg.py` a Python port of the segmenter (with pad before, pad after, minimum edge and minimum length options), `dec.py` the segment decode cache, `lib.py` the scoring and the paired bootstrap |
| `make_results.sh` | Rebuilds the tables in `results/` from the decode cache; it runs no ASR |
| `throwaway_cli_patch.diff` | A patch of `src/model.cpp` for measurement only. It is not part of the product. It adds three environment variables to `parakeet-cli` (dump the VAD probabilities, decode a given list of segments, print the words of each slice) |
| `results/` | The output of `make_results.sh`, plus `validate_cli_vs_harness.txt` |

Some tables (`results/sdi.txt`, `results/per_detector_default_table.md`) were made by one-off
commands on the same cache; their scripts were not kept, so `make_results.sh` does not rebuild
them. The other files in `results/` are rebuilt by it.

## How to regenerate

Environment variables: `PK_WORK` is the work directory (default: the current directory; it holds
`data/`, `probs/`, `cache/` and `tmp/`), `PK_CLI` is the patched `parakeet-cli`
(default `$PK_WORK/build/examples/cli/parakeet-cli`), `PK_GGUF` is a directory with
`ultra-q8_0.gguf`, `redux-keep.gguf` (the Redux head, packed), `tdt-0.6b-v3-q8_0.gguf` and
`silero-vad-f16.gguf`. Python packages: `numpy soundfile jiwer librosa datasets`.

```
git apply throwaway_cli_patch.diff                  # in a scratch checkout; build parakeet-cli from it
export PK_WORK=/path/to/work PK_CLI=/path/to/patched/parakeet-cli PK_GGUF=/path/to/models
python3 scripts/fetch.py $PK_WORK/data              # TED-LIUM long-form talks and LibriSpeech test-clean (streamed)
python3 scripts/mkcorpus.py $PK_WORK/data           # speech in noise sets, noise inserts, manifest.json (fixed seeds)
python3 scripts/probe.py 4                          # VAD probabilities of every file, per detector
python3 scripts/run.py T0,T0.1,T0.2,T0.3,T0.5,T1.0 talk,sinr,insert   # decode the segments of each variant (hours of CPU, resumes)
python3 scripts/validate.py                         # the harness against the patched CLI: segment bounds and text must match
sh make_results.sh                                  # tables into $PK_WORK/results
rm -r $PK_WORK/data                                 # audio
```

`scripts/variants.py` defines the variants: `Tx` is trim x seconds, `T0` the old cuts, `Px/y` a
pad of x before and y after, `Ex` trim only edges of at least x seconds, `Lx` trim only segments of
at least x seconds. `results/validate_cli_vs_harness.txt` shows 30 of 30 runs with the same segment
bounds and the same text as the CLI.
