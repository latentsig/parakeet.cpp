# Voice activity detection

parakeet.cpp has two voice activity detectors behind one API:

| Detector | Where it comes from | Frame | Input rates |
| --- | --- | --- | --- |
| VAD head | Inside the ASR GGUF of `moondream/parakeet-ultra` and `-redux` (see [ternary.md](ternary.md)) | 80 ms | 16 kHz (other rates are resampled) |
| Silero VAD | Its own small GGUF (`general.architecture` = `silero_vad`), [published](https://huggingface.co/mudler/parakeet-cpp-gguf) as `silero-vad-f32.gguf` and `silero-vad-f16.gguf` | 32 ms | 16 kHz and 8 kHz (other rates are resampled to 16 kHz) |

Both give one speech probability per frame. A shared segmenter
(`src/vad_segmenter.hpp`) turns the probabilities into speech regions or into
transcription segments. The segmenter takes the frame period as a parameter
(`SegmenterOpts::frame_sec`); only the option defaults differ per detector.

Accuracy and speed numbers for both detectors, with the method and the scripts to
repeat them, are in [vad-benchmarks.md](vad-benchmarks.md).

## Which detector to use

The Parakeet VAD head alone gives false alarms on audio without speech. On a speech-free
file the head calls about 99 percent of the frames speech (Ultra 99.4 percent, Redux 97.8
percent). Inside a file that has speech, over a 30 s stretch of noise at threshold 0.5,
the false-alarm frame rate was 17.7 percent for the Ultra head and 55 percent for the
Redux head. Silero showed 0 percent in both tests. This was measured on synthetic data:
LibriSpeech with added white and pink noise, see
[the experiment](vad-benchmarks.md#the-head-on-noise-only-audio).

This is not a bug in parakeet.cpp. An independent reference built from Hugging Face
transformers and the documented head structure matches `parakeet-cli vad --probabilities`
to a maximum probability difference of 2e-4 on F16 files. It is a property of the model.
The head judges each 80 ms frame by its level and texture relative to the average of the
file it is in. Steady loud noise, or a file that is only noise, sits at a logit of about
+1.5 to +2 (probability 0.8 to 0.9). Speech sits at +5 to +12 and pauses at -3 to -8. The
cards of the model say the head exists so that recordings can be cut at pauses into
segments of at most 30 s. What it was trained on is not stated. It is not a noise or
music rejector.

So use Silero, or Silero first with the head extending the boundaries, whenever long
stretches without speech are possible (always-on gates, calls, recordings with music or
noise). Use the head alone on audio that is known to be mostly speech (for example
before the transcription of recorded talks), or where its higher recall matters. If you
use the head alone and want fewer false alarms, raising the threshold to 0.9 removes most
noise false alarms at a cost of 1.5 to 2.6 F1 points on speech. Details, the noise types
that do and do not trigger it, and the mitigations we tested are in
[the benchmark page](vad-benchmarks.md#the-head-on-noise-only-audio).

For `transcribe --vad` the user-visible cost is small: some wasted decoding, and rarely a
hallucinated phrase. The 30 s hard cut of the segmenter can also keep a large part of a
long noise gap, even where the head called none of it speech. This is a segmenter
behaviour, not a head false alarm. On real recordings it is not the main cost after the
trim (see [the benchmark page](vad-benchmarks.md#the-30-s-hard-cut)).

**On real recordings** (36 VoxConverse recordings, 9 AMI far-field meetings and 14 AVA film
clips, 12.1 h with 9.5 h of labelled speech, and 2.6 h of music, noise and ESC-50 clips) the
picture is the same, with numbers:

- Frame F1 on speech, at the defaults: Silero 90.6, Ultra head 90.8, Redux head 92.8. The
  Redux head has the best default score and loses the least speech.
- On audio without speech, seconds called speech per hour: Silero 48, Ultra head 2174,
  Redux head 1685. The head calls most of a music or noise hour speech.
- On clean TED talks the choice of detector does not change the word error rate: all systems
  are within 0.17 percentage points. With 40 s of music, 30 s of noise and 40 s of vocal music
  inserted into each talk, the word error rate is 3.87 percent (Ultra) and 4.76 percent (Redux)
  when Silero cuts the audio, and 5.91 and 6.94 percent when the head does.

So the advice is:

- **Silero with its own defaults is the always-on gate.** Lowering its `threshold` to 0.2 to
  0.3 gives more recall (pooled F1 92.4 and 91.8 against 90.6) and at most 199 s of false
  alarm per hour on music (noise and ESC-50: 5 s/h or less).
- **The Redux head for long recordings that are mostly speech** (talks, meetings, interviews):
  the best default F1, the least speech lost, and the same word error rate as Silero.
- **Silero for recordings with long stretches of music or noise.**
- **The [run gate](#run-gate-opt-in) is an opt-in option for the heads** (0.92 to 0.96 for
  Redux; Ultra gains less). It removes most of the noise false alarms. It does not remove music
  and it is not a noise rejector.

An offline experiment also combined the two: Silero decides what is speech, and the head
only moves the edges. On synthetic clips it scored 0.75 to 1.13 F1 points above the best
single detector. That did not hold on real recordings: against the best single detector, tuned
the same way, the gain was 0.10 points for Ultra and 0.46 for Redux, with no change in word error
rate. It is not implemented in parakeet.cpp and there are no plans to add it. See
[Fusing Silero and the head](vad-benchmarks.md#fusing-silero-and-the-head-offline-experiment)
and [Real recordings](vad-benchmarks.md#real-recordings).

## Standalone VAD API

One call returns the speech regions of a clip as JSON, for either detector. Load
the model with `parakeet_capi_load`: it detects an ASR GGUF with a VAD head or a
Silero GGUF (`parakeet_capi_model_kind` is `PARAKEET_MODEL_KIND_VAD` for Silero).

```
char* parakeet_capi_vad_pcm_json(parakeet_ctx* ctx, const float* samples, int n_samples,
                                 int sample_rate, const char* options_json);
char* parakeet_capi_vad_path_json(parakeet_ctx* ctx, const char* wav_path, const char* options_json);

parakeet-cli vad --model <asr-with-vad-head.gguf | silero.gguf> --input audio.wav \
    [--mode speech|segments] [--probabilities] [--threshold F] [--min-pause SEC] \
    [--min-speech SEC] [--speech-pad SEC] [--max-segment SEC] [--trim SEC] [--run-gate P] [--threads N]
```

Both return NULL on error with the message in `parakeet_capi_last_error`, and the
caller frees the string with `parakeet_capi_free_string`. A model that is neither
kind fails with `model has no VAD head`.

The result has the same shape for both detectors:

```
{"mode":"speech","duration":23.605,"frame_sec":0.032,"backend":"cpu",
 "segments":[{"start":0.514,"end":5.534},{"start":6.882,"end":9.406}],
 "probabilities":[0.0123, ...]}      // only with "probabilities":true
```

Times are seconds with 3 decimals, on the timeline of the input. `frame_sec` is
0.080 for the head and 0.032 for Silero. `probabilities` has one value per frame
from time 0; the last Silero chunk is zero padded, as in Silero's own
`audio_forward`. `backend` names the compute device.

Options (`options_json` is NULL, "" or a flat JSON object; the CLI flags map to
the same keys). Unknown keys and out of range values are errors.

| Key | Meaning | VAD head | Silero |
| --- | --- | ---: | ---: |
| `threshold` | frame is speech when p >= threshold, in (0, 1] | 0.5 | 0.5 |
| `min_pause` | seconds; a silence this long separates regions, shorter gaps merge | 0.2 | 0.1 |
| `min_speech` | seconds; shorter speech runs are dropped | 0.1 | 0.25 |
| `speech_pad` | seconds >= 0, `speech` mode: widen each region on both sides | 0 | 0.03 |
| `max_segment` | seconds; cap in `segments` mode | 30 | 30 |
| `trim` | seconds >= 0; `segments` mode and the transcribe functions: shrink each cut to its speech plus this much on each side, 0 = keep the whole cut | 0.3 | 0.3 |
| `run_gate` | probability in [0, 1); drop a speech run whose median frame probability is below it, 0 = off (see [Run gate](#run-gate-opt-in)) | 0 | 0 |
| `mode` | `speech` or `segments` | `speech` | `speech` |
| `probabilities` | add the per frame probabilities | false | false |

Modes: `speech` is the smoothed speech regions for audio of any length. Gaps
shorter than 0.1 s are bridged, runs shorter than `min_speech` are dropped,
regions closer than `min_pause` merge, then each region is padded (and two
regions that would overlap meet in the middle of the gap). `segments` is the cut
that `transcribe --vad` decodes: pieces of at most `max_segment` seconds cut at
pauses, pieces without speech dropped, audio within the cap returned whole. Each
kept piece is then trimmed (see below).

### Trimming the segments

A piece cut from long audio used to carry everything between its two cut points
to the decoder, including the noise and silence the VAD had already flagged as
non-speech. That is where an ASR model invents words. Since this change each
piece shrinks to its first speech frame minus `trim` and its last speech frame
plus `trim` (default 0.3 s, never past the cut itself). Speech is the smoothed
mask, so the rules above still decide what counts as speech. Audio within the
cap is not cut and not trimmed. Word and token times are still relative to the
whole file. `trim` 0 (`--vad-trim 0`) gives the previous cuts exactly.

This is a change of default behaviour for `transcribe --vad`,
`parakeet_capi_transcribe_path_json_vad*` and the `segments` mode of the VAD
functions, for the head, for Silero and for VAD-only slices (they share the
segmenter). Transcripts of long audio can shift slightly. An enlarged measurement
(9 talks and about 375 noisy files) found no change in word error rate whose
interval excludes zero, for the Ultra head, the Redux head or Silero with TDT v3;
an earlier small run that read a loss for the Redux head was noise. On audio with
long noisy stretches the decoder sees much less noise, and with Silero the trim
stops whole sentences from being dropped on clean speech. Numbers:
[vad-benchmarks.md](vad-benchmarks.md#trimming-segments-and-the-word-filter).

## Run gate (opt-in)

The head calls steady noise and music speech, and the gate is a cheap check on that. A speech
run is a stretch of consecutive frames with `p >= threshold`. With `run_gate` set, a run is
dropped when the **median** of the probabilities of its frames is below the gate. Speech frames
sit at a logit of +5 to +12 (probability 0.99 and above). A steady noise run sits on a plateau
at a logit of +1.5 to +2 (0.8 to 0.9), with some higher peaks, so its median is lower than its
peak. Off by default (`run_gate` 0):
the output is then byte for byte what it was before the option existed.

```
parakeet-cli vad --model redux-vad.gguf --input a.wav --run-gate 0.92 [--mode segments]
parakeet-cli transcribe --model redux.gguf --input long.wav --vad --vad-run-gate 0.92
{"run_gate":0.92}     # in the options JSON of the vad and transcribe C functions
```

Definition (the same for every detector and both frame sizes):

- A run is a maximal stretch of frames with `p >= threshold`, found before any bridging. A gap that
  the segmenter later bridges is never inside a run, so a run does not borrow the probability of its
  neighbour.
- The median is over the probabilities of the frames of the run. For an even number of frames it is
  the mean of the two middle values. A run of one frame has its own probability as median.
- The run is kept when `median >= run_gate` and dropped when it is below. A median equal to the gate
  keeps the run.
- The gate acts first. The frames of a dropped run count as silence for bridging, `min_speech`, the
  pauses, the cuts and the trim. It applies to both modes (`speech` and `segments`), to every
  detector, and to the transcribe functions that take the VAD options.
- The value must be a number in [0, 1). Other values, and non-numbers, are errors, like every
  other key.

The gate is **offline only**. The streaming event tracker (`parakeet_capi_vad_stream_*`) decides frame
by frame and has no run median, so `parakeet_capi_vad_stream_begin` refuses a non-zero `run_gate`.
Audio of at most `max_segment` seconds is returned whole in `segments` mode without running the
segmenter, so the gate has no effect there (as for the trim).

Which value: **0.92 to 0.96 for the Redux head.** For Ultra the gate is weaker (see below). Silero
accepts the option, but its noise runs are rare, and the gate is meant for the heads.

What it does on real recordings (the full study is in
[vad-benchmarks.md](vad-benchmarks.md#real-recordings)). Frame F1 on speech, untuned, in percent:

| System | VoxConverse | AMI | AVA | Pooled |
| --- | ---: | ---: | ---: | ---: |
| Ultra head | 96.4 | 89.7 | 82.7 | 90.8 |
| Ultra head, gate 0.92 | 96.4 | 85.4 | 85.4 | 90.2 |
| Redux head | 97.3 | 92.2 | 85.4 | 92.8 |
| Redux head, gate 0.92 | 96.9 | 90.8 | 86.2 | 92.5 |
| Silero, own defaults | 96.4 | 86.7 | 84.7 | 90.6 |

The pooled cost of the gate at 0.92 is 0.6 points for Ultra and 0.3 for Redux. Both intervals
include zero. Seconds called speech per hour of audio without speech:

| System | Music | Noise | ESC-50 | Pooled |
| --- | ---: | ---: | ---: | ---: |
| Silero | 135 | 1 | 1 | 48 |
| Ultra head | 2823 | 1979 | 1605 | 2174 |
| Ultra head, gate 0.92 | 1625 | 398 | 395 | 826 |
| Redux head | 2123 | 1599 | 1236 | 1685 |
| Redux head, gate 0.92 | 644 | 23 | 131 | 269 |

The gate trades recall for false alarms. For the Redux head at 0.98 the music figure falls to 107
s/h, noise to 1 and ESC-50 to 15, but F1 on speech falls by 4.2 points. For the Ultra head at 0.98
the cost is 7.4 points and music is still 339 s/h. In `transcribe --vad` the effect is on the
decoded audio. Of one hour of non-speech with the trim at 0.3, the decoder gets this share:

| System (percent of the hour) | Music | Noise | ESC-50 |
| --- | ---: | ---: | ---: |
| Silero | 6 | 0.03 | 0.03 |
| Ultra head | 95 | 84 | 94 |
| Ultra head, gate 0.92 | 53 | 14 | 27 |
| Redux head | 91 | 82 | 94 |
| Redux head, gate 0.92 | 27 | 1.1 | 5.6 |

Word error rate: on clean TED talks every system is within 0.17 points, gate or not. On the three
talks with 40 s of music, 30 s of noise and 40 s of vocal music inserted, the gate recovers about
80 percent of the head's penalty against Silero: Ultra 5.91 (head), 4.24 (gate), 3.87 (Silero);
Redux 6.94, 5.22, 4.76.

What the gate does **not** do:

- **Music still triggers the head.** At 0.92 the Redux head still calls 644 s of an hour of music
  speech, and the Ultra head 1625 s. Music has a high median like speech does.
- **It is not a noise rejector.** It lowers the false alarms of the head on noise, but some remain:
  with the Redux head 23 s/h of noise and 131 s/h of ESC-50, with the Ultra head 398 and 395 s/h,
  against about 1 s/h for Silero. Use Silero where false alarms cost something.
- **It costs speech.** On AMI the F1 falls by 1.4 (Redux) and 4.3 (Ultra) points, and in `segments`
  mode on AVA the Redux gate loses 7.8 percent of the labelled speech (the head alone loses 0.2).
  It raises F1 on AVA, where the head over-detects, so the pooled cost is small.

Cross-check: the C++ gate gives the same regions as the Python gate of the study on 688
combinations of recording, detector and gate value, and the same speech seconds removed
(`scripts/vad_bench/real_recordings/verify_gate.py`).

## Word filter (opt-in)

A confidence filter can remove the words a model invents on noise. It is
post-processing of the decode: the model, the VAD and the cuts are unchanged, and
it is off by default (output is then byte for byte the same).

A word is dropped when the mean confidence of the words that start within
`local_radius` seconds of it (the word itself included, the same decode unit) is
below `min_local_conf`. A low confidence word between confident ones keeps a high
mean and stays; a word that stands alone, or among other low confidence words,
goes. A decode unit is the whole clip, or one VAD segment. `drop_punct_only` also
removes words that consist only of punctuation.

| Option | Meaning | Default |
| --- | --- | --- |
| `min_local_conf` | 0 to 1; 0 = off. 0.5 is the suggested value | 0 |
| `local_radius` | seconds, both sides | 5 |
| `drop_punct_only` | drop words with no letter or digit; recommended for CTC models | false |

```
parakeet-cli transcribe --model m.gguf --input a.wav --min-local-conf 0.5 \
    [--local-radius SEC] [--drop-punct-only] [--json] [--vad ...]

char* parakeet_capi_transcribe_path_json_with(parakeet_ctx* ctx, const char* wav_path,
                                              int decoder, const char* options_json);
```

The options JSON of `parakeet_capi_transcribe_path_json_with` takes the three
keys above. `parakeet_capi_transcribe_path_json_vad_with` takes them too, next to
the VAD keys (`trim` and `run_gate` included). With a filter on, the JSON document gets one more
member, `"guard":{"dropped_words":N}` (N is 0 when nothing was dropped), and the
dropped words are also removed from `text`, `words` and `tokens`.

Limits: 0.5 removes hallucinated words on Ultra, Redux and RNN-T models at no
cost in WER on clean speech. On v3 and on CTC models, and at higher thresholds,
it also removes real words (see the benchmark page). It does not save time: the
decoder still runs on everything it is given.

### Why the Silero defaults differ

The two defaults match what each model's authors recommend. For Silero these
are the values of `get_speech_timestamps` in its reference utilities: threshold
0.5, `min_speech_duration_ms` 250, `min_silence_duration_ms` 100,
`speech_pad_ms` 30. The segmenter code is shared. Silero closes a segment after
`min_silence` of silence and then drops it if it is shorter than
`min_speech`. The segmenter does the same in that order: it bridges gaps shorter
than 0.1 s, drops short runs, then merges at `min_pause`. The head keeps its
earlier defaults, so its output is byte for byte what it was before Silero
support (a test and a comparison of CLI output checked this). One known
difference: Silero's reference also lowers the threshold by 0.15 once speech has
started (hysteresis, `neg_threshold`); this segmenter has no hysteresis. If a
Silero clip flickers around 0.5, lower `threshold` or raise `min_pause`.

## Streaming Silero

A Silero context also gives a stream: feed audio in chunks of any size, read the
probabilities and the speech start and end events as they become known.

```
parakeet_vad_stream* parakeet_capi_vad_stream_begin(parakeet_ctx* vad, int sample_rate,
                                                    const char* options_json);
char* parakeet_capi_vad_stream_feed_json(parakeet_vad_stream* s, const float* pcm,
                                         int n_samples, int is_last);
int   parakeet_capi_vad_stream_reset(parakeet_vad_stream* s);
void  parakeet_capi_vad_stream_free(parakeet_vad_stream* s);
```

`sample_rate` is 16000 or 8000 (no resampling in a stream). The options are those
above, but `mode` must be `speech`. Each feed returns

```
{"frame_sec":0.032,"first_frame":120,
 "events":[{"type":"start","time":3.456},{"type":"end","time":5.120}],
 "probabilities":[0.01, ...]}          // only with "probabilities":true
```

- `events` are the starts and ends that became known in this call. An event is
  known late: a start once the speech lasted `min_speech`, an end once the silence
  lasted `min_pause`. The `time` is the speech boundary (with `speech_pad`), so it can
  be before the audio of the call. Times count from the start of the stream.
- `is_last` pads and scores the buffered partial chunk and closes an open region.
  After it, feed again only after `reset`.
- The probabilities do not depend on how the audio is split into calls. For
  `speech_pad` of at most `min_pause / 2` (true for both defaults) the events are
  the regions that the one shot `speech` mode gives for the whole audio
  (`VadEventTracker`, `test_vad_segmenter` checks this on random input).
- Errors come back as NULL and go to the last error of the Silero context. One
  stream is used from one thread at a time; different streams may run on
  different threads.

## Transcribing long audio with Silero

Any ASR model, with or without a VAD head (v3, Nemotron and others), can cut long
audio with Silero:

```
parakeet-cli transcribe --model tdt-0.6b-v3.gguf --input long.wav --vad --vad-model silero.gguf \
    [--vad-threshold F] [--vad-min-pause SEC] [--vad-min-speech SEC] [--vad-max-seg SEC] \
    [--vad-trim SEC]

char* parakeet_capi_transcribe_path_json_vad_with(parakeet_ctx* asr, parakeet_ctx* silero,
                                                  const char* wav_path, int decoder,
                                                  const char* options_json);
```

`--vad-model` implies `--vad`. The path uses the same segmenter in `segments`
mode (frame period 0.032 s) and the same decode of each segment as the head
path. Audio of at most 30 s is transcribed whole and the VAD does not run.
Passing NULL as the second argument uses the model's own head.

## VAD-only slice

The VAD head of Ultra and Redux reads only the log-mel front end, the subsampler
and the head, which is about 10 MB of the 213 MB to 1.4 GB file. `scripts/slice_vad_gguf.py`
copies exactly those tensors, byte for byte, into a small GGUF:

    python scripts/slice_vad_gguf.py ultra-q8_0.gguf ultra-vad.gguf

Nothing is requantized, so the VAD output is the same as the parent's. The slice
copies the `parakeet.encoder.*`, `parakeet.preprocessor.*` and `parakeet.vad.*`
keys. It sets `parakeet.arch` to `vad` (`general.architecture` stays `parakeet`),
and it records the parent in `parakeet.vad_only.parent_name`, `parent_file`,
`parent_sha256`, `parent_bytes` and `parent_arch`.

The `vad` marker is what the loader needs: a normal file has an encoder,
a decoder and a vocabulary, and the slice has none of them. `parakeet-cli vad`
and the `parakeet_capi_vad_*` functions accept the file (`Model::load_vad_only`).
`Model::load` refuses it with a log message, so every other call (transcribe,
diarize, stream) fails with "context holds a VAD-only model" or a load error.
Files that are not slices load exactly as before.

### Published slices

Two slices are published in [`mudler/parakeet-cpp-gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf).
They are cut out of Moondream's models, not trained here, and carry the same
CC-BY-4.0 license: credit Moondream and NVIDIA.

| File | Parent | Size (bytes) | SHA-256 |
|---|---|---:|---|
| [`redux-vad.gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/redux-vad.gguf) | parakeet-redux | 9,939,488 | `588e1d6e2ee5b6cdfd9ec5ea98dc0993d5bea498d9cc4ec8d6077041eef8a34f` |
| [`ultra-vad-q8_0.gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/ultra-vad-q8_0.gguf) | parakeet-ultra, Q8_0 | 6,007,328 | `8b891a4435e97438104ca07c72530d0c5fe62b986baee48b2dd4e1500c1d4758` |

    parakeet-cli vad --model redux-vad.gguf --input audio.wav

A slice cannot transcribe, diarize or stream.

The two Redux parents (packed ternary and dequantized F16) give slices with the same
tensors: the subsampler and the head are not ternary. The Ultra Q8_0 parent has a
Q8_0 final subsampler projection, so its slice is 6.0 MB; the F16 parents give
9.9 MB. A slice from a Q8_0 parent and a slice from an F16 parent give different
probabilities because their weights differ, not because of the slicing.

## ABI

All of this is additive. `parakeet_capi_abi_version` stays 10. A new symbol set
(`parakeet_capi_vad_stream_*`, `parakeet_capi_transcribe_path_json_vad_with`) and
the new model kind `PARAKEET_MODEL_KIND_VAD` (5) were added; no existing
signature or result changed. A caller that needs them checks for the symbols.

## Silero VAD model file

`pk::SileroVad` (`src/silero_vad.hpp`) runs the official Silero VAD model
(`snakers4/silero-vad`, version recorded in the GGUF) on the ggml backend. It is
a model-level component: it turns audio into one speech probability per 32 ms.
Turning probabilities into segments is the job of the segmenter described above.

Licence: the Silero VAD model and code are MIT licensed, copyright Silero Team
(<https://github.com/snakers4/silero-vad>). The GGUF files hold converted
weights of that model. Keep the licence and the attribution when you
redistribute them; the GGUF carries `general.license` and `general.url`.

Download `silero-vad-f32.gguf` or `silero-vad-f16.gguf` from
[mudler/parakeet-cpp-gguf](https://huggingface.co/mudler/parakeet-cpp-gguf).
They were converted from Silero VAD v6.2.3. To reproduce them, make the file from the official ONNX with `scripts/convert_silero_vad_to_gguf.py`
(see the schema in [conversion.md](conversion.md)). One file holds the 16 kHz
and the 8 kHz weights. The GGUF records the upstream version and the SHA-256 of
the ONNX file it came from. F32 is 2.2 MB, F16 is 1.3 MB.

## Using it

```cpp
std::string err;
auto vad = pk::SileroVad::load("silero-vad-f32.gguf", &err);   // nullptr + message on a bad file
auto stream = vad->new_stream(16000);                          // one per audio stream
std::vector<float> probs;
stream.process_chunk(pcm, n, &probs);   // any n; appends one value per completed chunk
stream.flush(&probs);                   // at the end: pads the last partial chunk
stream.reset();                         // start a new audio stream
```

- Input is mono float in [-1, 1] at 16000 or 8000 Hz. There is no resampler in
  the model; resample first.
- Time base: one chunk is 512 samples at 16 kHz and 256 samples at 8 kHz. Both are
  32 ms (`pk::kSileroFrameSec`). Probability `i` covers `[i * 0.032, (i + 1) * 0.032)`
  seconds. The model also sees the last 64 (16 kHz) or 32 (8 kHz) samples of the
  previous input, which `Stream` keeps.
- `process_chunk` accepts any number of samples and buffers them. Results do not
  depend on how the input is split into calls: they are equal bit for bit.
- `flush` zero-pads a partial last chunk, as the official `audio_forward` does.
  `probabilities(pcm, n, rate)` is the whole-clip form (`ceil(n / chunk)` values).
- Threads: the loaded model is read-only and shared. Each `Stream` holds its own
  state; use it from one thread at a time. Graph runs go through the process
  backend, which runs one graph at a time, so concurrent streams queue.

## Measured

On `tests/fixtures/speech.wav` padded with silence (310 chunks), against
onnxruntime 1.27.0 and the ONNX file of version 6.2.3:

| File | 16 kHz max diff | 8 kHz max diff |
| --- | ---: | ---: |
| F32 | 9e-7 | 2e-6 |
| F16 | 7e-4 | 3e-3 |

On one CPU thread, `tests/silero_vad_probe` measured about 40 us per chunk
(F32 or F16, 16 kHz or 8 kHz) on the development machine. That is one run on one
machine, not a benchmark. For benchmarks, see [vad-benchmarks.md](vad-benchmarks.md).

## Tests

- `test_silero_framer` and `test_silero_load_negative` need no model.
- `test_silero_vad` (label `model`) needs `PARAKEET_TEST_SILERO_GGUF`. It compares
  with `tests/fixtures/silero_vad_ref.txt` (limit 1e-4 for an F32 file, 5e-3 for
  a file whose name contains `f16`) and checks that streaming equals batch.
- `test_vad_segmenter` (no model) covers the 32 ms grid, the Silero defaults, the
  padding and the streaming event tracker. `test_vad_options` (no model) covers the
  option parser and the NULL and bad file paths of the C-API.
- `test_vad_run_gate` (no model) covers the run gate on synthetic probability streams at 80 ms and
  32 ms frames: runs above and below the gate, a low median with a high peak, bridged gaps, a
  one frame run, the boundary (a median equal to the gate keeps the run), the trim, and that gate 0 does not change
  the output. `test_vad_run_gate_model` (label `model`, the VAD-only slice, Ultra, Redux and Silero
  files) runs the gate on a clip with a noise stretch.
- `test_capi_vad_silero` (label `model`, `PARAKEET_TEST_SILERO_GGUF`) covers the C-API
  document at both rates, options, errors, threads and the stream.
- `test_transcribe_vad_silero` (label `model`, `PARAKEET_TEST_SILERO_GGUF` and
  `PARAKEET_TEST_GGUF`, an ASR model such as v3) checks that a long synthetic clip
  transcribed in Silero segments matches the plain transcript within a word
  difference of 0.15, and the C-API function.
- Regenerate the reference with `scripts/gen_silero_vad_ref.py silero_vad.onnx`
  (needs `onnxruntime`). The test clip is built from `speech.wav` at run time.
