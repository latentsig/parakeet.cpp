# Voice activity detection

## Silero VAD model file

`pk::SileroVad` (`src/silero_vad.hpp`) runs the official Silero VAD model
(`snakers4/silero-vad`, version recorded in the GGUF) on the ggml backend. It is
a model-level component: it turns audio into one speech probability per 32 ms.
Turning probabilities into segments is the job of a segmenter on top.

Licence: the Silero VAD model and code are MIT licensed, copyright Silero Team
(<https://github.com/snakers4/silero-vad>). The GGUF files hold converted
weights of that model. Keep the licence and the attribution when you
redistribute them; the GGUF carries `general.license` and `general.url`.

Make the file from the official ONNX with `scripts/convert_silero_vad_to_gguf.py`
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
- Regenerate the reference with `scripts/gen_silero_vad_ref.py silero_vad.onnx`
  (needs `onnxruntime`). The test clip is built from `speech.wav` at run time.
