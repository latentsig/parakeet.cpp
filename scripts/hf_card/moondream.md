### ultra and redux (Moondream)

Source: [moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) and [moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) by [Moondream](https://huggingface.co/moondream), derived from NVIDIA's [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) · TDT transducer (FastConformer), multilingual, with a voice-activity head (`transcribe --vad`) · heads: TDT · license: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

| File | Variant | Size | Runs on |
|---|---|---:|---|
| `ultra-f16.gguf` | F16 | 1441.9 MB | any backend |
| `ultra-q8_0.gguf` | Q8_0 | 941.5 MB | any backend |
| `redux-packed.gguf` | packed ternary encoder | 213.3 MB | CPU only, offline only |
| `redux-f16.gguf` | F16, dequantized | 1441.9 MB | any backend |
| `redux-q8_0.gguf` | Q8_0, dequantized | 941.5 MB | any backend |

> There is no NeMo reference for these models, so there is no WER column. Each file was checked to decode the `tests/fixtures/speech.wav` clip to the expected sentence. See [ternary.md](https://github.com/mudler/parakeet.cpp/blob/master/docs/ternary.md) for the measurements.

- **Ultra** has ordinary F16 weights and runs on any backend, like the other v3 files.
- **Redux packed** keeps the encoder linear layers as ternary weights (-1, 0 or +1 times a per-group scale) and runs a native CPU kernel. It does not run on GPU backends or in streaming mode, and parakeet.cpp refuses to load it there. For GPU use, take `redux-f16.gguf` or `redux-q8_0.gguf`.
- **Redux F16 and Q8_0** are dequantized: the ternary weights were expanded to ordinary weights and then stored as F16 or Q8_0. They are not packed.
- **Changes:** these files are converted here, not trained. Nothing was trained or fine-tuned by the parakeet.cpp project. The models were trained by NVIDIA (the base) and Moondream (Ultra and Redux).

```bash
huggingface-cli download mudler/parakeet-cpp-gguf redux-packed.gguf --local-dir models/
build/examples/cli/parakeet-cli transcribe --model models/redux-packed.gguf --input audio.wav
# Long audio: cut at pauses with the model's voice-activity head (offline only).
build/examples/cli/parakeet-cli transcribe --model models/ultra-q8_0.gguf --input long.wav --vad
```
