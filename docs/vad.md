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
| `mode` | `speech` or `segments` | `speech` | `speech` |
| `probabilities` | add the per frame probabilities | false | false |

Modes: `speech` is the smoothed speech regions for audio of any length. Gaps
shorter than 0.1 s are bridged, runs shorter than `min_speech` are dropped,
regions closer than `min_pause` merge, then each region is padded (and two
regions that would overlap meet in the middle of the gap). `segments` is the cut
that `transcribe --vad` decodes: pieces of at most `max_segment` seconds cut at
pauses, pieces without speech dropped, audio within the cap returned whole.

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
    [--vad-threshold F] [--vad-min-pause SEC] [--vad-min-speech SEC] [--vad-max-seg SEC]

char* parakeet_capi_transcribe_path_json_vad_with(parakeet_ctx* asr, parakeet_ctx* silero,
                                                  const char* wav_path, int decoder,
                                                  const char* options_json);
```

`--vad-model` implies `--vad`. The path uses the same segmenter in `segments`
mode (frame period 0.032 s) and the same decode of each segment as the head
path. Audio of at most 30 s is transcribed whole and the VAD does not run.
Passing NULL as the second argument uses the model's own head.

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
machine, not a benchmark.

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
