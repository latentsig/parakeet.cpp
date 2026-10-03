### silero-vad (Silero Team)

Source: [snakers4/silero-vad](https://github.com/snakers4/silero-vad) v6.2.3 by the [Silero Team](https://github.com/snakers4) · a small voice-activity detector, not a speech recognizer · 8 kHz and 16 kHz in one file · license: [MIT](https://github.com/snakers4/silero-vad/blob/master/LICENSE)

| File | Variant | Size |
|---|---|---:|
| `silero-vad-f32.gguf` | F32 | 2.2 MB |
| `silero-vad-f16.gguf` | F16 weights, widened to F32 at load | 1.3 MB |

- The files are converted from the official ONNX model with `scripts/convert_silero_vad_to_gguf.py` in parakeet.cpp. The GGUF records the source version and the ONNX sha256. They were converted here, not trained.
- They need a parakeet.cpp build that has the standalone VAD API. That code is not in a release yet, so check the parakeet.cpp repository before relying on it.
- Use it as a stand-alone detector (`parakeet-cli vad --model silero-vad-f16.gguf --input audio.wav`) or to cut long audio before transcribing with any model, including those that have no VAD head (`parakeet-cli transcribe --model tdt-0.6b-v3-q8_0.gguf --input long.wav --vad --vad-model silero-vad-f16.gguf`).
- Probabilities match onnxruntime to about 1e-6 (F32) and 1e-3 (F16) on the test clips.
