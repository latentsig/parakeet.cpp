# Real-recording VAD study

Scripts and result tables behind the section
[Real recordings](../../../docs/vad-benchmarks.md#real-recordings) of
`docs/vad-benchmarks.md` and the run gate of [docs/vad.md](../../../docs/vad.md#run-gate-opt-in).
No audio, probability file or model is committed, only the scripts and the small result tables
in `results/` (about 100 kB).

The study compares Silero, the Ultra head and the Redux head on real recordings:
59 recordings with human speech labels (12.1 h of audio, 9.5 h of labelled speech) and 27 files
without speech (2.6 h). `results/recordings.json` lists every recording with its domain, split,
duration and labelled speech seconds.

| Domain | Source | Recordings | Audio |
| --- | --- | ---: | ---: |
| `vox` | VoxConverse (`diarizers-community/voxconverse`), dev and test splits, every 8th recording | 36 | 4.4 h |
| `ami` | AMI far-field (`diarizers-community/ami`, config `sdm`), validation every 3rd and test every 2nd | 9 | 4.2 h |
| `ava` | AVA-Speech film clips with human labels (`nccratliri/vad-human-ava-speech`), chosen by an MD5 order | 14 | 3.5 h |
| `music` | MUSAN music (`corypaik/musan`, config `music`) | 16 | 0.9 h |
| `noise` | MUSAN noise (`corypaik/musan`, config `noise`) | 6 | 1.0 h |
| `esc` | ESC-50 (`ashraq/esc50`), every 4th clip, joined into long files | 5 | 0.7 h |

Tuning and test are disjoint by recording: VoxConverse dev and AMI validation tune, VoxConverse test and
AMI test are held out, AVA and the non-speech files split by the parity of an MD5 of the id. Only 20
speech recordings are held out (AMI: 3 meetings), so the held-out intervals are wide.
The TED-LIUM long-form talks of the WER part come from `../fetch_ted_talks.py`.

## Setup

All paths are arguments or environment variables. Run from any directory.

| Variable | Meaning | Default |
| --- | --- | --- |
| `VAD_REAL_ROOT` | work directory: `data/<domain>/` (audio and labels), `probs/`, `results/` | current directory |
| `PARAKEET_CLI` | a `parakeet-cli` built from master (any build with `vad --probabilities`) | `$VAD_REAL_ROOT/parakeet-cli-clean` |
| `VAD_REAL_MODELS` | directory with `silero-vad-f16.gguf`, `ultra-vad-q8_0.gguf`, `redux-vad.gguf` (the VAD-only slices; for `wer_dec.py` also `ultra-q8_0.gguf` and `redux-packed.gguf`) | `$VAD_REAL_ROOT/models` |
| `HF_HOME` | Hugging Face cache for the fetch scripts | `$VAD_REAL_ROOT/hfhome` |

Python packages: `numpy numba soundfile librosa datasets huggingface_hub`; the fetch of AVA also needs `ffmpeg`.

## Steps

1. Fetch the data. `fetch_diar.py NAME CONFIG SPLITS K CAP_HOURS OUT_DIR` keeps the recordings whose index
   modulo K equals K//2 in each split, up to the hour cap and writes a 16 kHz mono WAV and a JSON
   with the speaker turns. `fetch_ava.py OUT_DIR N` downloads N clips. `fetch_nonspeech.py music|noise|esc OUT_DIR ...`
   writes the non-speech files. The exact recordings of the published run are in `results/recordings.json`.
2. `dump_probs.py` runs `parakeet-cli vad --probabilities` for every recording and detector and stores the
   per-frame probabilities (`probs/*.npy`) next to the CLI's own speech regions.
3. `verify_native.py` checks that the Python segmenter replica in `vr.py` gives the CLI's regions;
   `verify_seg.py` does the same for the `segments` mode.
4. `eval_frames.py` scores every system on a 10 ms grid (frame precision, recall, F1) and `report_frames.py`,
   `report_options.py`, `report_fusion_grid.py` and `extras.py` print the tables of `results/frames.md`,
   `options.md`, `fusion_grid.md` and `extras.md` (confidence intervals are bootstrap over recordings).
5. `seg_eval.py` and `seg_report.py` measure what the segmenter sends to the decoder (`results/seg.md`).
6. WER: `wer_plan.py` writes the segment list of every system; `wer_dec.py` decodes those segments;
   `wer_report.py` prints `results/wer.md`. `mk_composite.py` builds the composite recordings (three TED
   talks with 40 s of music, 30 s of noise and 40 s of vocal music inserted at pauses) and the plans with
   argument `comp` give `results/wer_comp.md`. `wer_plan_forced.py` and `wer_report.py forced` give
   `results/wer_forced.md`, the cost of a forced cut.
   `wer_dec.py` needs a throwaway `parakeet-cli` patched to decode a given list of segments (it reads them
   from the file named by the environment variable `PK_SEGMENTS` and writes the words to `PK_SEGOUT`). That
   patch is not part of the repository, so this step cannot be repeated from the repository alone.
7. `verify_gate.py` checks the C++ run gate against the Python gate (`vr.gate_frames`) on the stored
   probabilities. Build `gate_segtool.cpp` first; the command is in the docstring of the script. On the
   688 comparisons of the published run (all recordings, three detectors, two or three gates each) the regions
   were identical, and the speech seconds the Redux head loses at 0.92 matched to the millisecond.

## How the gate is defined here

`vr.gate_frames(p, thr, med)` finds every run of consecutive frames with `p >= thr` in the detector's own
frames (80 ms or 32 ms), before any bridging, and keeps the run when the median of those frames is at least `med`.
This is the definition of `SegmenterOpts::run_gate` in `src/vad_segmenter.hpp`.

## Results

`results/frames.md` frame F1 of every system, tuned and untuned, with paired differences (sections A to F);
`options.md` option sweeps for Silero and the heads; `fusion_grid.md` the fusion rule grid; `extras.md` the
collar test and music by source; `seg.md` what the decoder receives (trim, cut policies); `wer.md`,
`wer_comp.md` and `wer_forced.md` word error rates; `tuned.json` the parameters chosen on the tuning split.
