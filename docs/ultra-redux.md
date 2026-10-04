# Moondream Parakeet Ultra and Redux

This page holds the model notes for the two Moondream derivatives of Parakeet. The
runtime details are in other pages: the ternary kernels and their speed in
[ternary.md](ternary.md), the VAD head and Silero in [vad.md](vad.md), and the
measurements in [vad-benchmarks.md](vad-benchmarks.md).

[moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) and
[moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) are Moondream's
post-trained (Ultra, F16) and ternary-encoder (Redux) derivatives of NVIDIA's
[parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). Both are released under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). They are HF safetensors, converted with
`../scripts/convert_hf_parakeet_to_gguf.py`. They are not part of the NeMo-validated set above: there is
no NeMo baseline for them, so parity is transcript-level against our own v3 path (see
[`parity.md`](parity.md)), and the GGUFs are published in
[mudler/parakeet-cpp-gguf](https://huggingface.co/mudler/parakeet-cpp-gguf): `ultra-f16.gguf`, `ultra-q8_0.gguf`, `redux-packed.gguf`
(packed ternary), `redux-f16.gguf` and `redux-q8_0.gguf` (dequantized). Sizes and SHA-256 sums are in
[`models/MANIFEST.md`](../models/MANIFEST.md).

The models were trained by NVIDIA (the base) and Moondream (Ultra and Redux). parakeet.cpp only
converts and quantizes the weights; nothing is trained or fine-tuned here. A dequantized Redux file
(`--ternary dequant`, the converter default) holds ordinary F16 or Q8_0 weights expanded from the
ternary ones.

- Redux packs the encoder as ternary weights: a 213 MB GGUF, 6.8x smaller than F16. It runs on CPU
  only and offline only. On x86 with AVX-512 VNNI it reaches median RTF 75.6 per utterance on
  LibriSpeech-100 (8 threads) against 46.1 for the same model in F16; on a single 180 s clip the
  gain is about 10 percent; WER on the 100 LibriSpeech utterances is 1.96 percent. See
  [`ternary.md`](ternary.md). SIMD kernels exist for x86-64 with AVX2 or AVX-512 VNNI and
  aarch64 with dotprod; MSVC builds, Windows on ARM and aarch64 without dotprod use a slow scalar
  kernel (about 1 GMAC/s), and the load logs a warning. The packed file also stays resident next to
  the repacked planes, so memory use is more than the file size.
- Both carry a voice-activity head, used by `transcribe --vad` to cut long audio at pauses. Speech
  is a probability of at least 0.5; pauses of at least 0.2 s are candidate cuts, segments are at most
  30 s, and segments without speech are dropped. On long-form clips it does not change WER
  meaningfully. Details and measurements: [`ternary.md`](ternary.md).
- The same head runs on its own, without transcribing: `parakeet-cli vad`, or
  `parakeet_capi_vad_pcm_json` / `parakeet_capi_vad_path_json` from the C-API. They return speech
  segments (start and end in seconds) as JSON, and optionally the per-frame probabilities. A model
  without the head fails with `model has no VAD head`. See [`vad.md`](vad.md).
- Silero VAD (MIT, 32 ms frames, 16 kHz and 8 kHz) runs from its own small GGUF ([download](https://huggingface.co/mudler/parakeet-cpp-gguf)) through the same
  functions, as a stream (`parakeet_capi_vad_stream_*`), and as the cutter for any ASR model:
  `parakeet-cli transcribe --vad --vad-model silero.gguf`. See [`vad.md`](vad.md).
- VAD-only slices of Ultra and Redux (6 to 10 MB, `redux-vad.gguf` and `ultra-vad-q8_0.gguf` in the same repo) hold just the head and its front end. They run `vad` and the `parakeet_capi_vad_*` calls and cannot transcribe. See [`vad.md`](vad.md).
- The head alone gives false alarms on audio without speech: on speech-free noise it calls about
  99 percent of the frames speech (Ultra 99.4, Redux 97.8), and over a 30 s noise stretch inside a
  file with speech the false-alarm frame rate was 17.7 percent for Ultra and 55 percent for Redux,
  against 0 percent for Silero (synthetic LibriSpeech with added noise). Prefer Silero as the
  always-on gate or when the audio can have long non-speech stretches; use the head on audio known
  to be mostly speech, or where its higher recall matters. An offline experiment that lets Silero
  decide and the head move the edges (not implemented here) is in
  [`vad-benchmarks.md`](vad-benchmarks.md#fusing-silero-and-the-head-offline-experiment).
- Accuracy, speed and size of both detectors, with the method and the scripts to repeat
  them: [`vad-benchmarks.md`](vad-benchmarks.md).
