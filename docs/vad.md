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
behaviour, not a head false alarm, and it has not been fixed.

An offline experiment also combined the two: Silero decides what is speech, and the head
only moves the edges. It scored 0.75 to 1.13 F1 points above the best single detector on
the synthetic clips and gave no false alarms on noise. It is not implemented in
parakeet.cpp. See
[Fusing Silero and the head](vad-benchmarks.md#fusing-silero-and-the-head-offline-experiment).

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
    [--min-speech SEC] [--speech-pad SEC] [--max-segment SEC] [--threads N]
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
segmenter). On talks, transcripts of long audio can shift slightly (a word WER
cost of about 0.1 point in our runs); on audio with long noisy stretches the
decoder sees much less noise. Numbers: [vad-benchmarks.md](vad-benchmarks.md#trimming-segments-and-the-word-filter).

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
the VAD keys (`trim` included). With a filter on, the JSON document gets one more
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
- `test_capi_vad_silero` (label `model`, `PARAKEET_TEST_SILERO_GGUF`) covers the C-API
  document at both rates, options, errors, threads and the stream.
- `test_transcribe_vad_silero` (label `model`, `PARAKEET_TEST_SILERO_GGUF` and
  `PARAKEET_TEST_GGUF`, an ASR model such as v3) checks that a long synthetic clip
  transcribed in Silero segments matches the plain transcript within a word
  difference of 0.15, and the C-API function.
- Regenerate the reference with `scripts/gen_silero_vad_ref.py silero_vad.onnx`
  (needs `onnxruntime`). The test clip is built from `speech.wav` at run time.
