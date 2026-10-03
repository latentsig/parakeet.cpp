# VAD benchmarks

This page lists every voice activity detection (VAD) measurement made for
parakeet.cpp so far: how each one was built, what it measures, the numbers, and
how to run it again. For the API and the model files, see [vad.md](vad.md).

Read the caveats before you quote a number. Every reference here is coarse, the
machine was shared, and some runs were not on a quiet machine. A difference that
is smaller than the noise is not claimed.

The detectors:

| Name | What it is |
| --- | --- |
| Silero | Silero VAD 6.2.3 (MIT), 32 ms frames, from its own small GGUF (F32 or F16) |
| Ultra Q8_0 | VAD head of `moondream/parakeet-ultra`, GGUF in Q8_0, 80 ms frames |
| Redux packed | VAD head of `moondream/parakeet-redux`, packed ternary GGUF, 80 ms frames |
| whisper.cpp Silero | Silero VAD inside whisper.cpp (`whisper_vad_*`), used as a comparison |
| ONNX Silero | The official Silero ONNX file run with onnxruntime (CPU, one thread), used as the reference |

## Summary

Accuracy is the frame level F1 (percent) against a coarse reference, see
[Metrics](#metrics). Speed is audio seconds per wall second ("x real time") on a
300 s clip. Higher is faster.

| Detector | F1, clean synthetic | F1, pink noise 0 dB | F1, TED-LIUM talks | Speed | Peak memory | File size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Silero (matched settings) | 94.1 | 91.7 | 91.8 | about 660x on 1 thread, `parakeet-cli vad`, includes process start and model load | 60 MB (`parakeet-cli vad`, F32 and F16) | 2.2 MB (F32), 1.3 MB (F16) |
| Silero (its own defaults) | 94.4 | 91.9 | 92.1 | not timed separately | same as above | same as above |
| Ultra Q8_0 head | 95.0 | 93.4 | 93.8 | 216x, 8 threads | pending, see below | 941.5 MB (the whole ASR model) |
| Redux packed head | 95.4 | 94.3 | 94.2 | 221x, 8 threads | pending, see below | 213.3 MB (the whole ASR model) |
| whisper.cpp Silero | 94.1 | not run | not run | in process: 686x on 1 thread | 36 MB (test harness) | 0.9 MB |
| ONNX Silero (onnxruntime) | 94.0 | not run | not run | in process: 449x on 1 thread, Python loop | 104 MB | 2.3 MB |

How to read it:

- The synthetic and TED columns come from the
  [heads against Silero](#parakeet-heads-against-silero) benchmark. The whisper.cpp
  and ONNX rows come from the [Silero parity](#silero-in-whispercpp-and-in-parakeetcpp)
  benchmark, which uses a different synthetic corpus (120 clips, 3 conditions). The
  clean F1 of "Silero" and "whisper.cpp Silero" are therefore not from the same clips.
  Compare rows only inside one benchmark.
- "Matched settings" means Silero ran with the same segmentation settings as the
  heads (threshold 0.5, minimum speech 100 ms, minimum silence 200 ms, no padding).
  See [Metrics](#metrics).
- The head speed is the speed of a whole encoder pass on 8 threads, because the head
  reads the encoder output. The four head builds that were timed (Ultra Q8_0 216x,
  Ultra F16 218x, Redux packed 221x, Redux dequantized F16 222x) are within 3% of
  each other. This is a different job from the one Silero does, so the speed columns
  of Silero and the heads are not a like for like comparison of the models. They show
  the cost of using each one as a stand alone gate.
- Memory of the head as a stand alone VAD is **pending**: it needs a measurement of
  the head without the rest of the ASR model, see
  [Slice only head](#slice-only-head-pending).
- File size of the heads is the size of the whole ASR GGUF the head lives in. A
  stand alone head file is part of the pending measurement.

### Machine and build

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 9 class (Zen 5), 16-core desktop part; the OS shows 20 logical CPUs, one thread per core |
| Memory | 84 GB |
| OS and compiler | Linux 6.8, GCC 13.3, CMake 3.28 |
| ggml | commit e705c5fe (the submodule of this repository), `GGML_NATIVE=ON`, Release |
| parakeet.cpp | the VAD code of PR 83 (merged as `6165e3d`); see the per benchmark build notes below |
| whisper.cpp | commit `60c0be6ac8fa71b1a2ae2dd938a31a34a508e774` (version 1.9.4), CPU only |
| Load | the machine was shared with other jobs; load is given per benchmark |

Build commits per benchmark:

| Benchmark | parakeet.cpp build |
| --- | --- |
| Heads against Silero | a development build of the VAD branch; the commit was not recorded (unverified) |
| Silero in whisper.cpp and in parakeet.cpp | `c1a8391102d34074f48e758c441758327d2465b7`, the VAD branch before its squash into `6165e3d` |
| Long talks, batched decode | three builds, called base, b0 and b1: master before PR 81, PR 81 (`cbabaf5`), PR 82 (`9fdf3a1`). The result files do not record the commits; the mapping is from the PR 82 description (unverified) |
| Timing re-run | `6165e3d`, see [Timing re-run](#timing-re-run) |

## Metrics

All VAD output is reduced to speech regions (start and end in seconds).

- **Frame level precision, recall, F1.** The regions of the detector and of the
  reference are turned into a speech mask on a 10 ms grid. Counts of true positive,
  false positive, false negative and true negative frames are summed over all clips
  of a group, then precision, recall and F1 are computed from the sums. Pooled, not
  an average of per clip scores.
- **Boundary error.** For each reference start (and end), the distance in ms to the
  nearest detected start (end). Reported as median and 90th percentile, in ms, plus
  the miss rate: the share of reference boundaries with no detected boundary within
  1 s. "All conditions" pools start and end errors.
- **Agreement.** Two detectors compared with each other: F1 of the two speech masks,
  Cohen's kappa, and IoU (intersection over union) of the masks.
- **Probability parity.** Per frame, the absolute difference of the speech
  probability against onnxruntime on the official ONNX file. Also the share of
  frames where the decision at 0.5 flips.
- **Matched settings.** The segmentation step is not the same code in every system,
  and the defaults differ. For a fair model comparison, Silero in the first
  benchmark runs the `silero-vad` package (`get_speech_timestamps`) with threshold
  0.5, `min_speech_duration_ms` 100, `min_silence_duration_ms` 200,
  `speech_pad_ms` 0. These are the head defaults. Silero's reference code also
  lowers the threshold by 0.15 once speech has started (hysteresis); the parakeet.cpp
  segmenter has no hysteresis (see [vad.md](vad.md)). "Silero (its own defaults)" uses
  the package defaults (minimum speech 250 ms, minimum silence 100 ms, padding 30 ms).
  The heads ran with their own defaults, which are the matched values.

## Parakeet heads against Silero

### Data

Synthetic clips. Built from LibriSpeech test-clean (public). Every clip has five
utterances in a random order. Each utterance is trimmed at the first and last 10 ms
frame whose RMS is above 1% of the peak frame RMS. The utterances are joined with
gaps of 0.3 to 2.0 s (uniform), with 0.5 to 1.5 s before the first and after the
last. The gaps hold Gaussian noise at a standard deviation of 0.001. The reference
speech region of an utterance is the whole trimmed utterance, so short pauses inside
an utterance count as speech.

Noise is added to the whole clip at a signal to noise ratio (SNR) measured against
the power of the speech regions: white or pink, at 20, 10, 5 and 0 dB, plus a clean
condition. That gives 9 conditions of 24 clips: **216 clips**, **1080 reference
regions**. The speech layout of clip k is the same in every condition.

How the 24 clips were chosen is partly unverified. The script that cut the
LibriSpeech subset for this run was not saved. `scripts/vad_bench/fetch_data.py`
takes every 13th utterance of test-clean, which is the scheme of the Silero parity
corpus, and the numbers below were measured on a subset of that kind. A rebuild with
the script will give the same kind of corpus, not the same bytes, and the numbers
will differ slightly.

TED-LIUM talks. Three talks from `distil-whisper/tedlium-long-form` (test split),
3178.6 s in total. Which three is not recorded in the result files (unverified). The
reference is built from word timestamps of the 0.6B v3 F16 ASR model run on 30 s
pieces: words closer than 0.4 s are merged into one region. This reference is
derived from an ASR model and is coarser than the synthetic one. It is also not
independent of the Parakeet family. The reference speech fraction is 91.9%. One more
talk was left out of the run by a name filter; the reason was not recorded.

### Results: synthetic, threshold 0.5

Cells are precision / recall / F1, in percent. "Silero M" is Silero with matched
settings, "Silero D" with its own defaults.

| Condition | Silero M | Silero D | Ultra Q8_0 | Redux packed |
| --- | --- | --- | --- | --- |
| clean | 99.1 / 89.7 / 94.1 | 98.7 / 90.5 / 94.4 | 98.2 / 92.0 / 95.0 | 98.4 / 92.7 / 95.4 |
| white 20 dB | 99.4 / 88.4 / 93.5 | 99.0 / 89.2 / 93.9 | 98.8 / 89.9 / 94.1 | 98.7 / 91.7 / 95.1 |
| white 10 dB | 99.3 / 88.0 / 93.3 | 98.9 / 89.0 / 93.7 | 98.9 / 88.3 / 93.3 | 98.2 / 91.9 / 94.9 |
| white 5 dB | 99.2 / 87.8 / 93.1 | 98.8 / 88.5 / 93.4 | 98.9 / 87.2 / 92.7 | 97.4 / 91.9 / 94.6 |
| white 0 dB | 99.2 / 85.8 / 92.0 | 98.8 / 86.7 / 92.4 | 98.3 / 87.9 / 92.8 | 96.4 / 92.3 / 94.3 |
| pink 20 dB | 99.4 / 88.4 / 93.6 | 99.1 / 89.3 / 93.9 | 98.8 / 90.8 / 94.6 | 98.6 / 92.0 / 95.2 |
| pink 10 dB | 99.3 / 88.0 / 93.3 | 99.0 / 89.0 / 93.7 | 99.0 / 88.8 / 93.6 | 98.1 / 92.2 / 95.1 |
| pink 5 dB | 99.4 / 87.6 / 93.1 | 99.0 / 88.5 / 93.5 | 99.0 / 88.4 / 93.4 | 97.4 / 92.6 / 94.9 |
| pink 0 dB | 99.2 / 85.2 / 91.7 | 98.8 / 85.9 / 91.9 | 96.3 / 90.8 / 93.4 | 95.5 / 93.2 / 94.3 |

What the table supports: the three detectors are within about 1.5 F1 points of each
other at 5 dB and above, and up to 2.6 points apart at 0 dB. Silero has the higher precision and the lower recall. Both
heads have the higher recall. In the loudest noise (0 dB) the heads keep their F1
better than Silero (pink 0 dB: Silero M 91.7, Ultra 93.4, Redux 94.3). The recall
of all detectors is below 94% because the reference counts pauses inside utterances
as speech, so the absolute recall values are not a measure of missed speech.

Boundary error, in ms (median / 90th percentile, miss rate), start then end:

| Condition | Silero M | Silero D | Ultra Q8_0 | Redux packed |
| --- | --- | --- | --- | --- |
| clean | S 171/527 (0.0%) E 121/366 (0.0%) | S 162/499 (0.0%) E 137/343 (0.0%) | S 123/460 (0.0%) E 168/325 (0.0%) | S 152/459 (0.0%) E 151/328 (0.0%) |
| white 10 dB | S 249/526 (0.0%) E 116/371 (0.0%) | S 232/500 (0.0%) E 129/357 (0.0%) | S 211/470 (0.0%) E 147/472 (0.8%) | S 163/466 (0.8%) E 152/389 (0.8%) |
| white 0 dB | S 262/571 (0.8%) E 117/459 (2.5%) | S 237/526 (0.8%) E 142/451 (1.7%) | S 162/461 (0.0%) E 149/494 (2.5%) | S 202/460 (0.8%) E 149/404 (1.7%) |
| pink 10 dB | S 248/526 (0.0%) E 106/384 (0.0%) | S 232/502 (0.0%) E 117/343 (0.0%) | S 226/479 (0.0%) E 136/445 (0.0%) | S 176/445 (0.0%) E 148/340 (0.0%) |
| pink 0 dB | S 261/570 (0.0%) E 145/411 (3.3%) | S 248/530 (0.0%) E 150/408 (3.3%) | S 180/457 (2.5%) E 162/470 (2.5%) | S 200/457 (1.7%) E 160/428 (1.7%) |

All conditions pooled:

| System | Median | 90th percentile | Start median | End median | Miss rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Silero M | 143 | 501 | 237 | 116 | 0.4% |
| Silero D | 149 | 484 | 232 | 130 | 0.3% |
| Ultra Q8_0 | 154 | 465 | 181 | 149 | 0.6% |
| Redux packed | 153 | 437 | 170 | 149 | 0.4% |

Boundary errors of 100 to 250 ms are large next to the 32 and 80 ms frame sizes.
Most of this comes from the reference (trimmed utterances next to short gaps), so
only the ordering of start errors is informative: the heads place starts about 50
to 70 ms closer to the reference than Silero M does. The signed start error median
is 237 ms for Silero M, 232 for Silero D, 162 for Ultra and 147 for Redux (the
detector starts later than the reference). The signed end error median is 3, 20, -10
and 54 ms. Segments per reference region: Silero M 1.96, Ultra 1.92, Redux 1.76
(a value above 1 means the detector splits utterances at their inner pauses).

Threshold sweep, pooled over clean, white 10 dB, white 0 dB and pink 10 dB
(precision / recall / F1):

| System | threshold 0.3 | threshold 0.5 | threshold 0.7 |
| --- | --- | --- | --- |
| Silero M | 98.5 / 90.3 / 94.3 | 99.2 / 87.9 / 93.2 | 99.5 / 86.2 / 92.4 |
| Ultra Q8_0 | 97.1 / 92.4 / 94.7 | 98.6 / 89.3 / 93.7 | 99.0 / 86.2 / 92.1 |
| Redux packed | 94.8 / 95.0 / 94.9 | 97.7 / 92.3 / 94.9 | 98.9 / 90.0 / 94.2 |

Agreement with Silero M on all synthetic clips: Ultra F1 96.6, kappa 0.882, IoU 93.4.
Redux F1 96.3, kappa 0.865, IoU 92.8. Ultra against Redux: F1 97.2, kappa 0.896.

### Results: TED-LIUM talks

Frame level on 3178.6 s, with the ASR derived reference (speech fraction 91.9%):

| System | Precision | Recall | F1 | Speech fraction found |
| --- | ---: | ---: | ---: | ---: |
| Silero M | 97.3 | 86.8 | 91.8 | 82.0% |
| Silero D | 97.4 | 87.4 | 92.1 | not recorded |
| Ultra Q8_0 | 97.1 | 90.7 | 93.8 | 85.8% |
| Redux packed | 97.1 | 91.5 | 94.2 | 86.6% |

Agreement: Ultra and Silero M F1 97.0, kappa 0.813; Redux and Silero M F1 96.9,
kappa 0.801; Ultra and Redux F1 98.8, kappa 0.912. The "Silero D" speech fraction
was not printed in the run output.

The ordering is the same as on the synthetic clips: equal precision, the heads
have the higher recall. The gap in F1 between Silero M and the heads is 2 to 2.5
points. Because the reference comes from a Parakeet model, a bias in favor of the
Parakeet heads is possible and was not measured.

### Results: speed of the heads and of Silero ONNX

300 s of one talk (the first 300 s of the 1249 s talk), 4 repetitions, load average
5.8 to 6.2 during the run. Silero ran through the `silero-vad` Python package
(ONNX, one thread). The heads ran with 8 threads.

| System | Speed, x real time |
| --- | ---: |
| Silero ONNX, 1 thread | 393 |
| Ultra Q8_0, 8 threads | 216 |
| Ultra F16, 8 threads | 218 |
| Redux packed, 8 threads | 221 |
| Redux dequantized F16, 8 threads | 222 |

The run printed one speed per system. The result file does not say whether it is the
best or the median of the 4 repetitions (unverified). The speeds that the TED script
printed during the accuracy run (load average 6 to 42) are not valid and are not
used. Thread pinning was not recorded for this run.

### Caveats

- References are coarse. The synthetic reference counts pauses inside utterances as
  speech and uses an energy trim. The TED reference comes from ASR word times.
- Absolute precision and recall depend on that reference. Compare detectors with
  each other, on the same clips.
- Differences of less than about one F1 point between two detectors in one cell are
  not claimed. There are no confidence intervals; one corpus was used.
- Speed numbers: see the load notes above. Detectors with different thread counts
  are not a model comparison.
- The build commit of this run was not recorded.

## Silero in whisper.cpp and in parakeet.cpp

This benchmark asks whether the Silero port in parakeet.cpp gives the same result
and the same speed as the Silero VAD that ships in whisper.cpp.

### Setup

- **Models.** `silero_vad.onnx` (version 6.2.3, SHA-256
  `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`);
  parakeet.cpp GGUF F32 (`1398e5ce...`) and F16 (`8160489282...`) made with
  `scripts/convert_silero_vad_to_gguf.py` from that ONNX file; whisper.cpp model
  `ggml-silero-v6.2.3-ggml.bin` (`abd79739...`), converted from the same ONNX with the
  whisper.cpp conversion script, and the model of the whisper.cpp model repository
  `ggml-silero-v6.2.0.bin` (`2aa269b7...`). The full list of hashes and versions is in
  [`silero_versions_and_sha256.txt`](../scripts/vad_bench/results/silero_versions_and_sha256.txt).
- **Reference.** onnxruntime on the official ONNX file, with the official 64 sample
  context, one thread.
- **whisper.cpp access.** A small test program (`scripts/vad_bench/wvad.cpp`) over the
  public `whisper_vad_*` functions: probabilities, and the segments of the whisper.cpp
  C++ code with its defaults and with matched parameters. It is not part of
  whisper.cpp.
- **Corpus.** 120 synthetic clips (40 clips of 5 utterances, in clean, white 10 dB and
  pink 10 dB), built with the same recipe as above from 200 LibriSpeech test-clean
  utterances (every 13th). Plus the first 600 s of one TED-LIUM talk.
- **Segmentation.** To compare only the models, the same Python code turns every
  probability stream into segments: threshold 0.5, negative threshold 0.35, minimum
  speech 100 ms, minimum silence 200 ms, padding 0 ("matched post"). The own defaults
  of whisper.cpp and parakeet.cpp are also shown.

### Probability parity against onnxruntime

All 120 clips and the 600 s talk, one value per 32 ms chunk:

| System | Max abs diff | Mean abs diff | Frames > 0.05 off | Decision agreement at 0.5 | Flipped frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| whisper.cpp, converted 6.2.3 | 0.7373 | 0.00224 | 0.54% | 99.826% | 309 |
| whisper.cpp, repository 6.2.0 | 0.7373 | 0.00224 | 0.54% | 99.826% | 309 |
| parakeet.cpp F32 | 0.0001 | 0.00002 | 0.00% | 99.999% | 2 |
| parakeet.cpp F16 | 0.0036 | 0.00005 | 0.00% | 99.997% | 5 |

whisper.cpp (converted 6.2.3) against parakeet.cpp F32: mean abs diff 0.00224, 311
flipped frames. The two whisper.cpp model files give the same result (0 flipped
frames between them). Mean abs diff against onnxruntime, first chunk of each clip /
the rest: whisper.cpp 0.00088 / 0.00212, parakeet.cpp F32 0.00002 / 0.00002, F16
0.00005 / 0.00005. On the 600 s talk alone, the number of flipped frames against
onnxruntime is 57 of 18750 for whisper.cpp, 0 of 18750 for parakeet.cpp F32 and 1
for F16. The cause of the whisper.cpp difference was not investigated.

### Segment quality on the synthetic clips

Precision / recall / F1 in percent, 40 clips per condition, same reference as above:

| System (settings) | clean | white 10 dB | pink 10 dB | all |
| --- | --- | --- | --- | --- |
| onnxruntime (matched post) | 99.3 / 89.3 / 94.0 | 99.4 / 87.8 / 93.2 | 99.5 / 87.4 / 93.1 | 99.4 / 88.2 / 93.5 |
| whisper.cpp (matched post) | 99.3 / 89.4 / 94.1 | 99.4 / 87.8 / 93.3 | 99.5 / 87.4 / 93.0 | 99.4 / 88.2 / 93.5 |
| parakeet.cpp F32 (matched post) | 99.3 / 89.3 / 94.0 | 99.4 / 87.8 / 93.2 | 99.5 / 87.4 / 93.1 | 99.4 / 88.2 / 93.5 |
| parakeet.cpp F16 (matched post) | 99.3 / 89.3 / 94.0 | 99.4 / 87.8 / 93.2 | 99.5 / 87.4 / 93.1 | 99.4 / 88.2 / 93.5 |
| whisper.cpp (its own defaults) | 98.9 / 90.3 / 94.4 | 99.2 / 89.0 / 93.8 | 99.3 / 88.6 / 93.6 | 99.1 / 89.3 / 94.0 |
| whisper.cpp (its C++ code, matched parameters) | 99.3 / 89.4 / 94.1 | 99.4 / 87.8 / 93.3 | 99.5 / 87.4 / 93.0 | 99.4 / 88.2 / 93.5 |
| parakeet.cpp F32 (its own defaults) | 99.1 / 89.3 / 93.9 | 99.3 / 87.6 / 93.1 | 99.4 / 87.3 / 93.0 | 99.3 / 88.1 / 93.3 |

Boundary error against the reference, all conditions pooled (start median / 90th
percentile, end median / 90th percentile, miss rate): onnxruntime, whisper.cpp,
parakeet.cpp F32 and F16 with matched post all give 241/553 and 108/396, 0.2%.
whisper.cpp with its own defaults gives 209/521, 124/367, 0.0%. parakeet.cpp with its
own defaults gives 211/523, 116/392, 0.0%.

Agreement of the speech masks (matched post), 120 clips pooled (F1 / kappa / IoU):

| Pair | F1 | kappa | IoU |
| --- | ---: | ---: | ---: |
| whisper.cpp vs parakeet.cpp F32 | 99.92 | 0.9969 | 99.84 |
| whisper.cpp vs onnxruntime | 99.92 | 0.9969 | 99.84 |
| parakeet.cpp F32 vs onnxruntime | 100.00 | 1.0000 | 100.00 |
| parakeet.cpp F16 vs F32 | 100.00 | 0.9999 | 100.00 |
| whisper.cpp 6.2.3 vs 6.2.0 file | 100.00 | 1.0000 | 100.00 |

On the 600 s talk (no reference), speech fraction found: onnxruntime 79.1%,
whisper.cpp 78.8%, parakeet.cpp F32 79.1%. whisper.cpp against parakeet.cpp F32:
F1 99.75, kappa 0.9882.

What this supports: with the same post processing, whisper.cpp Silero and parakeet.cpp
Silero give the same segment quality on these clips (all within 0.1 F1 point). The
difference between whisper.cpp and parakeet.cpp in their own defaults (94.0 against
93.3 F1 on all clips) comes from their different default settings, not from the model.
The F16 file is not measurably worse than F32 here.

### Speed

Each job ran on one pinned core, on a 300 s clip (the first 300 s of the 600 s talk; the
clip file was not kept, so this is assumed from the preparation script),
serialized with a lock. A run started only when the mean number of other runnable
tasks over 10 s was below 4, or the 1-minute load average was below 5. The machine has
20 logical CPUs and was never idle, so the 1-minute load average at the start of the
runs was between 4.2 and 15.9, and the mean runnable count was 1.8 to 5.5. These are
not quiet machine numbers. Wall time is the process wall time, including
process start and model load for the command line tools.

| Job | Runs | Wall time, median (min to max), s | In process time, median, s | x real time (median) | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| whisper.cpp `detect_speech` only (test program) | 5 | 0.456 (0.446 to 0.473) | 0.438 | 686 | 36 MB |
| `whisper-vad-speech-segments` | 5 | 0.452 (0.444 to 0.457) | not applicable | 664 | 28 MB |
| `parakeet-cli vad`, F32 | 5 | 0.455 (0.425 to 0.464) | not applicable | 659 | 60 MB |
| `parakeet-cli vad`, F16 | 5 | 0.451 (0.436 to 0.553) | not applicable | 666 | 60 MB |
| onnxruntime, Python, 1 thread | 3 | 0.806 (0.738 to 1.241) | 0.668 | 449 | 104 MB |

The x real time column is 300 s divided by the in process time when there is one,
else by the wall time. The four Silero command line jobs are within the spread of
their repeated runs (0.43 to 0.55 s), so the numbers do not show a speed difference
between whisper.cpp and parakeet.cpp. The onnxruntime figure is a Python loop that
calls the session once per 512 sample chunk. It is not a tuned onnxruntime number.

whisper.cpp with `tiny.en`, one thread, same 300 s clip, end to end: 15.14 s
without VAD, 19.68 s with the Silero VAD (3 runs each, load average 4.6 to 8.3). The
ASR part dominates. Why the run with VAD is slower was not investigated.

### Caveats

- The load notes above apply. The runs were on a shared machine.
- One clip and one core. No claim is made about other lengths, other CPUs, or more
  threads.
- The onnxruntime run had 3 repetitions. The other jobs had 5.
- The measured whisper.cpp is a source build at the commit above, CPU only.

## Long talks: batched decode for VAD segments

PR 82 decodes the VAD segments in groups of up to 16, using the exact batched
decoder of PR 81. Each segment is still encoded alone. This measures that change, not
the VAD itself.

Data: four TED-LIUM long-form talks of 18 to 25 minutes, merged to one file each.
Models: Ultra Q8_0 and Redux packed. Command: `parakeet-cli transcribe --vad --json
--threads 8`, pinned to 8 cores. Three builds, interleaved, 5 runs each:
**120 runs**. The build order was rotated for each repetition. Before each run the
script waited until the mean number of other runnable tasks over 10 s was below 4.
The mean number of other runnable tasks at the gate was 0 to 1.5, but the 1-minute
load average at the start of the runs was 2.5 to 18.0 (it includes the earlier runs of
the same job and other jobs), see
`scripts/vad_bench/results/longform_b1_loadlog.txt` and the per run values in
`longform_b1_results.tsv`. The 1-minute load average was therefore often above 5.
Wall time is the time of the whole command, including model load.

Median wall time in seconds over 5 runs:

| Talk (length, s) | Model | Segments | base | PR 81 | PR 82 | PR 82 vs PR 81 | Peak RSS PR 82 vs PR 81 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BillGates (1505.8) | Ultra Q8_0 | 59 | 44.2 | 43.7 | 40.8 | 1.072x | +1.7% |
| BillGates | Redux packed | 62 | 28.1 | 27.7 | 25.9 | 1.071x | +2.2% |
| AimeeMullins (1249.0) | Ultra Q8_0 | 46 | 36.9 | 36.9 | 34.8 | 1.058x | +1.7% |
| AimeeMullins | Redux packed | 46 | 23.4 | 23.7 | 21.9 | 1.079x | +2.5% |
| JaneMcGonigal (1167.6) | Ultra Q8_0 | 50 | 35.9 | 34.9 | 34.0 | 1.029x | +1.4% |
| JaneMcGonigal | Redux packed | 47 | 24.1 | 23.1 | 23.0 | 1.006x | +2.2% |
| DanielKahneman (1095.5) | Ultra Q8_0 | 44 | 32.6 | 33.1 | 32.9 | 1.006x | +1.7% |
| DanielKahneman | Redux packed | 42 | 21.4 | 21.5 | 21.2 | 1.013x | +2.2% |

The average of the eight speed ratios is 1.042x (range 1.006x to 1.079x). Peak
memory of the whole transcription is 1.4 to 2.5 percent higher with PR 82, 30 to 40
MB on a base of 1.4 to 2.0 GB. The transcript (JSON output) has the same hash for all
three builds in all 8 talk and model pairs.

Caveats: the gain is small and several of the ratios are inside the spread of the
5 runs (some single runs of the same configuration differ by more than 3%); do not
claim more than a small gain. Only CPU and x86 were measured. The memory numbers are
for the whole transcription, not for the VAD.

## Silero parity with the reference (PR 83)

This table is also in [vad.md](vad.md). Test clip: `tests/fixtures/speech.wav` padded
with silence (310 chunks), against onnxruntime 1.27.0 and the ONNX file of version
6.2.3. Maximum probability difference:

| File | 16 kHz | 8 kHz |
| --- | ---: | ---: |
| F32 | 9e-7 | 2e-6 |
| F16 | 7e-4 | 3e-3 |

One run of `tests/silero_vad_probe` on one CPU thread measured about 40 us per chunk
for F32 and F16 at both rates. That is one run on one machine. It is not a benchmark
and is not used above.

## Slice only head (pending)

Running the head needs the whole ASR GGUF today. A measurement of a head file that
holds only the head and the encoder part it needs ("slice") is being made by another
piece of work. It will give the load time, the peak memory and the speed of the head
as a stand alone detector.

| Quantity | Value |
| --- | --- |
| Peak memory, head alone | pending |
| Load time, head alone | pending |
| Speed, head alone | pending |
| File size, head alone | pending |

No number is given here until that measurement is published with its method.

## Timing re-run

Not run yet when this text was written.

## When to use which

This is limited to what the numbers above support.

- **Stand alone gate, any ASR model, small files.** Silero. Its GGUF is 1.3 to 2.2
  MB, the command line tool needs about 60 MB, and it ran at about 660x real time on
  one thread on a shared machine. It gave frame level F1 of 91.7 to 94.4 on the
  synthetic data and 91.8 to 92.1 on the TED talks.
- **The Parakeet head when Ultra or Redux is already loaded.** The head costs no extra
  model and its scores are close to Silero in the same tests: within about 1.5 F1 points
  on the synthetic clips at 5 dB and above (Ultra was 0.4 lower in white noise at 5 dB),
  2 to 2.5 points higher on the TED talks, and 1.7 to 2.6 points higher at 0 dB pink
  noise. It has the higher recall and the lower precision. On these clips its start times are closer to the reference. It
  is not a cheap stand alone detector today: it needs the encoder, its speed is about
  220x on 8 threads, and its memory as a stand alone VAD is not yet measured.
- **Silero from whisper.cpp or from parakeet.cpp.** On the 120 clips and the 600 s talk
  the two give the same segments (mask F1 99.92) and the same speed within the noise of
  a shared machine. Choose by the rest of your stack, not by quality or speed. The
  probabilities of parakeet.cpp match onnxruntime more closely (mean abs diff 0.00002,
  2 flipped frames) than those of whisper.cpp do (0.00224, 309 flipped frames), but this
  did not change the segment scores.
- **Not supported by these numbers:** any claim about GPUs, ARM, other languages,
  music or non speech noise, streaming latency, or the effect of the VAD on WER.

## How to reproduce

The scripts are in [`scripts/vad_bench/`](../scripts/vad_bench/), with a README that
lists the inputs, the commands and the Python packages. No audio and no model files
are committed. In short:

1. Fetch the public audio: `fetch_data.py`, `fetch_ted_talks.py`.
2. Heads against Silero: `compare_synthetic.py` then `analyze_synthetic.py`;
   `ted_reference.py` then `compare_ted.py`; `speed_heads.py`.
3. Silero parity: `make_clips.py`, build `wvad.cpp` against whisper.cpp,
   `silero_collect.py`, `silero_analyze.py`; speed with `speed_silero.py`.
4. Long talks: `longform_b1.sh`.

Small result files of the runs on this page (tables, per run timings and load logs)
are in `scripts/vad_bench/results/`. The raw per clip predictions are not committed;
a re-run regenerates them.
