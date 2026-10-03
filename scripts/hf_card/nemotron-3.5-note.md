> This model runs both offline and as a cache-aware streaming model, and it is the only one here that takes a target language. Pass `--lang <locale>` (for example `en-US`, `de-DE`, `es-ES`, `ja-JP`), or leave it at the default `auto` to let the model detect the language. The WER column is measured against NeMo, offline, on the languages en, de and auto; streaming and offline output also match NeMo for en, de, es, ja-JP and auto. See [parity.md](https://github.com/mudler/parakeet.cpp/blob/master/docs/parity.md). The small prompt layers, the LSTM and the feature extractor stay F32 in every quantization.

```bash
huggingface-cli download mudler/parakeet-cpp-gguf nemotron-3.5-asr-streaming-0.6b-f16.gguf --local-dir models/
build/examples/cli/parakeet-cli transcribe --model models/nemotron-3.5-asr-streaming-0.6b-f16.gguf --input audio.wav --lang de-DE
```
