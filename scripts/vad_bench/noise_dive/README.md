# Why the Parakeet VAD head fires on noise

Scripts and small results of the root-cause study described in
[../../../docs/vad-benchmarks.md](../../../docs/vad-benchmarks.md#the-head-on-noise-only-audio).
No audio, model, `.npz` or `.pkl` file is committed. A re-run regenerates them.

## Inputs (set as environment variables)

| Variable | Meaning |
| --- | --- |
| `FUSION_DIR` | `scripts/vad_bench/fusion` with `data/libri.npy` and `data/spk.npy` built by `prep_libri.py` (default `../fusion`) |
| `NOISE_DIVE_WORK` | working directory with `ref/` (reference weights as `<name>_weights.npz`), `hfmeta/` (`<name>.config.json`) and the CLI build |
| `PARAKEET_CLI` | path to `parakeet-cli` |
| `GGUF_ROOT` | directory with `gguf/` and `vad-card/` GGUF files |
| `SILERO_ONNX` | path to `silero_vad.onnx` (used by `e_build.py`, `e2_talk.py`) |

Run the scripts from this directory. Python packages: numpy, torch, soundfile,
transformers (with the Parakeet classes), onnxruntime, matplotlib.

## Files

- `vadlib.py`: independent reference of the head path built on Hugging Face transformers.
  `sig.py` and `corpus.py`: signal generators and the speech clips.
- `a_faith.py`, `cli_check.py`, `a2_variants.py`: the reference against the CLI; head variants (`res_A*`, `res_cli_check.txt`).
- `b1` to `b10`: amplitude, signal types, normalisation, regression, level and share of noise (`res_B*`).
- `d_prep.py`, `d_run.py`, `d_analyze.py`, `d_probs.py`, `d_mit.py`: the 60 s block in a real talk with `transcribe --vad` (`res_D*`).
  `res_D_silero.txt` holds the Silero segment check, done by hand from its segment file.
- `e_build.py`, `e_eval.py`, `e2_talk.py`: the mitigations (`res_E*`).
- `fig_*.png`: four figures (logit histograms, false alarms against level, level and variation of the stretch, offset and scale probe).
- The `res_*.md` and `res_*.txt` files are the results quoted in the docs.
