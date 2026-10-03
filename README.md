# parakeet.cpp

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team**, the folks behind LocalAI, the open-source AI engine that runs any model (LLMs, vision, voice, image, video) on any hardware, no GPU required.

[![Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/mudler/parakeet-cpp-gguf)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![LocalAI](https://img.shields.io/badge/LocalAI-Run_Locally-orange)](https://github.com/mudler/LocalAI)

parakeet.cpp is a C++17 inference port of NVIDIA's [NeMo](https://github.com/NVIDIA-NeMo/NeMo) Parakeet speech-recognition models, built on [ggml](https://github.com/ggml-org/ggml). It gives you fast, dependency-light automatic speech recognition on CPU (and on GPU through ggml's backends), with no Python runtime needed at inference time.

It covers all the offline Parakeet families (CTC, RNNT, TDT, and hybrid TDT-CTC, in 0.6B/1.1B/110M sizes, English plus multilingual v3), each validated at WER 0 against NeMo on every published checkpoint. It also does **cache-aware streaming with end-of-utterance (EOU) detection** for `parakeet_realtime_eou_120m-v1`, where the streaming transcript matches NeMo's cache-aware streaming byte for byte. And it supports the **multilingual, prompt-conditioned streaming model** `nvidia/nemotron-3.5-asr-streaming-0.6b` (40+ locales): pass a target language with `--lang <locale>` (default `auto`) and both the offline and the cache-aware streaming transcripts match NeMo per language at WER 0. The full coverage matrix lives in `docs/parity.md`.

It's faster than NeMo's PyTorch runtime on both CPU and GPU, with byte-identical transcripts. The full numbers, methodology, and all the plots are in [benchmarks/BENCHMARK.md](benchmarks/BENCHMARK.md).

<p align="center">
  <a href="benchmarks/BENCHMARK.md"><img src="benchmarks/plots/speedup.png" width="49%" alt="CPU speedup vs NeMo (RTFx ratio per dtype)"></a>
  <a href="benchmarks/BENCHMARK.md"><img src="benchmarks/plots/gpu_speedup.png" width="49%" alt="GPU speedup vs NeMo on the NVIDIA GB10"></a>
</p>

It also runs circles around whisper.cpp on the same audio: the 110M Parakeet is faster than whisper base.en and far faster than large-v3-turbo, while the larger Parakeets match or beat whisper's accuracy (see [benchmarks/BENCHMARK.md](benchmarks/BENCHMARK.md)).

<p align="center">
  <a href="benchmarks/BENCHMARK.md"><img src="benchmarks/plots/vs_whisper.png" width="88%" alt="parakeet.cpp vs whisper.cpp RTFx on CPU and GPU"></a>
</p>

---

## Supported models

Every model below is validated at WER 0 against NeMo and published as GGUF (f16, q8_0, q6_k, q5_k, q4_k) in the single collection repo [mudler/parakeet-cpp-gguf](https://huggingface.co/mudler/parakeet-cpp-gguf). Convert any of them yourself with `scripts/convert_parakeet_to_gguf.py`. The per-model parity matrix is in [docs/parity.md](docs/parity.md).

| Model | Type | Size | Notes | Source |
| ----- | ---- | ---- | ----- | ------ |
| [parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) | hybrid TDT+CTC | 110M | English, the small anchor checkpoint | NVIDIA |
| [parakeet-ctc-0.6b](https://huggingface.co/nvidia/parakeet-ctc-0.6b) | CTC | 0.6B | English | NVIDIA |
| [parakeet-rnnt-0.6b](https://huggingface.co/nvidia/parakeet-rnnt-0.6b) | RNNT | 0.6B | English | NVIDIA |
| [parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | TDT | 0.6B | English | NVIDIA |
| [parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | TDT | 0.6B | multilingual (25 European languages) | NVIDIA |
| [parakeet-ctc-1.1b](https://huggingface.co/nvidia/parakeet-ctc-1.1b) | CTC | 1.1B | English | NVIDIA |
| [parakeet-rnnt-1.1b](https://huggingface.co/nvidia/parakeet-rnnt-1.1b) | RNNT | 1.1B | English | NVIDIA |
| [parakeet-tdt-1.1b](https://huggingface.co/nvidia/parakeet-tdt-1.1b) | TDT | 1.1B | English | NVIDIA |
| [parakeet-tdt_ctc-1.1b](https://huggingface.co/nvidia/parakeet-tdt_ctc-1.1b) | hybrid TDT+CTC | 1.1B | English | NVIDIA |
| [parakeet_realtime_eou_120m-v1](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) | RNNT, streaming | 120M | cache-aware streaming with end-of-utterance detection (`--stream`) | NVIDIA |
| [nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | RNNT, streaming | 0.6B | multilingual (40+ locales), prompt-conditioned, offline and cache-aware streaming, pick a language with `--lang` (default `auto`). OpenMDW-1.1 | NVIDIA |
| [parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) | TDT | 0.6B | Moondream's post-trained derivative of parakeet-tdt-0.6b-v3, with a VAD head. CC-BY-4.0. Not NeMo-validated, see below | Moondream, from NVIDIA |
| [parakeet-redux](https://huggingface.co/moondream/parakeet-redux) | TDT | 0.6B | Moondream's ternary-encoder derivative of parakeet-tdt-0.6b-v3, with a VAD head. CPU only. CC-BY-4.0. Not NeMo-validated, see below | Moondream, from NVIDIA |


### Moondream Ultra and Redux (not yet published)

[moondream/parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) and
[moondream/parakeet-redux](https://huggingface.co/moondream/parakeet-redux) are Moondream's
post-trained (Ultra, F16) and ternary-encoder (Redux) derivatives of NVIDIA's
[parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3). Both are released under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). They are HF safetensors, converted with
`scripts/convert_hf_parakeet_to_gguf.py`. They are not part of the NeMo-validated set above: there is
no NeMo baseline for them, so parity is transcript-level against our own v3 path (see
[`docs/parity.md`](docs/parity.md)), and no GGUFs are published yet.

The models were trained by NVIDIA (the base) and Moondream (Ultra and Redux). parakeet.cpp only
converts and quantizes the weights; nothing is trained or fine-tuned here. A dequantized Redux file
(`--ternary dequant`, the converter default) holds ordinary F16 or Q8_0 weights expanded from the
ternary ones.

- Redux packs the encoder as ternary weights: a 213 MB GGUF, 6.8x smaller than F16. It runs on CPU
  only and offline only. On x86 with AVX-512 VNNI it reaches median RTF 75.6 per utterance on
  LibriSpeech-100 (8 threads) against 46.1 for the same model in F16; on a single 180 s clip the
  gain is about 10 percent; WER on the 100 LibriSpeech utterances is 1.96 percent. See
  [`docs/ternary.md`](docs/ternary.md). SIMD kernels exist for x86-64 with AVX2 or AVX-512 VNNI and
  aarch64 with dotprod; MSVC builds, Windows on ARM and aarch64 without dotprod use a slow scalar
  kernel (about 1 GMAC/s), and the load logs a warning. The packed file also stays resident next to
  the repacked planes, so memory use is more than the file size.
- Both carry a voice-activity head, used by `transcribe --vad` to cut long audio at pauses. Speech
  is a probability of at least 0.5; pauses of at least 0.2 s are candidate cuts, segments are at most
  30 s, and segments without speech are dropped. On long-form clips it does not change WER
  meaningfully. Details and measurements: [`docs/ternary.md`](docs/ternary.md).
- The same head runs on its own, without transcribing: `parakeet-cli vad`, or
  `parakeet_capi_vad_pcm_json` / `parakeet_capi_vad_path_json` from the C-API. They return speech
  segments (start and end in seconds) as JSON, and optionally the per-frame probabilities. A model
  without the head fails with `model has no VAD head`. See [`docs/vad.md`](docs/vad.md).
- Silero VAD (MIT, 32 ms frames, 16 kHz and 8 kHz) runs from its own small GGUF through the same
  functions, as a stream (`parakeet_capi_vad_stream_*`), and as the cutter for any ASR model:
  `parakeet-cli transcribe --vad --vad-model silero.gguf`. See [`docs/vad.md`](docs/vad.md).
---

## Performance

parakeet.cpp is faster than NeMo's PyTorch runtime on every Parakeet model, on both CPU and GPU, and the transcripts come out byte-identical (WER 0 vs NeMo). Full methodology, all 10 models, quantization tradeoffs, and plots are in [`benchmarks/BENCHMARK.md`](benchmarks/BENCHMARK.md).

### See it run

The same clip fed to parakeet.cpp and to NeMo's own PyTorch runtime on the same GPU. The output comes out byte-for-byte identical, parakeet.cpp just gets there first (slowed down so the sub-100ms race is watchable):

![parakeet.cpp vs NeMo on GPU: identical output, parakeet.cpp finishes first](benchmarks/media/gpu_duel.gif)

The same race on CPU, against NeMo's own PyTorch runtime: [parakeet.cpp vs NeMo on CPU](benchmarks/media/cpu_nemo_duel.mp4) (about 1.5x faster, still byte-for-byte identical). And vs whisper.cpp turbo, same accuracy and far less compute: [on GPU](benchmarks/media/gpu_whisper_duel.mp4) (about 12x faster) and [on CPU](benchmarks/media/cpu_duel.mp4) (about 27x faster).

CPU numbers (20-core x86, vs NeMo PyTorch-CPU, LibriSpeech test-clean, threads=8; RTFx is audio-seconds over processing-seconds, so higher is faster):

| dtype | size vs f32 | speedup vs NeMo | accuracy |
| ----- | ----------- | --------------- | -------- |
| f32   | 100%        | 1.11 to 1.69x (median 1.40x) | WER 0, byte-identical to NeMo |
| f16   | 57%         | up to 1.70x     | near-lossless |
| q8_0  | 37%         | up to 1.86x     | near-lossless |
| q4_k  | 26%         | n/a             | small, monotonic WER cost |

Peak RAM is also roughly 2x lower than NeMo, and lower still once quantized.

GPU numbers (NVIDIA GB10, Grace-Blackwell, vs NeMo-GPU in the `nvcr.io/nvidia/nemo` container, since NeMo can't run on the host's torch/CUDA stack directly): parakeet.cpp wins on all 10 models, with a median of 1.25x and up to 4.3x on the large TDT/hybrid models. NeMo's TDT greedy decode isn't CUDA-graph accelerated and ours is a lean C++ loop, which is most of that gap. The log-mel front end runs on the GPU via a ggml DFT-matmul graph; the CPU path is unchanged.

---

## Pre-built binaries

Every [release](https://github.com/mudler/parakeet.cpp/releases) ships pre-built `parakeet-cli` bundles, so there is no need to compile from source:

| Platform | Variants |
| -------- | -------- |
| Linux x64 | cpu, vulkan, cuda |
| Linux arm64 | cpu |
| macOS arm64 | metal |
| macOS x64 | cpu |
| Windows x64 | cpu, vulkan, cuda |

The cuda bundles target Turing (sm_75) and newer, including Blackwell. On Linux the CUDA runtime libraries are bundled in the tarball; on Windows download the `cudart-parakeet-bin-win-cuda-x64.zip` asset alongside the binary zip unless you already have the CUDA toolkit installed. The vulkan binaries need the Vulkan loader on the system (`libvulkan1` on Debian/Ubuntu; on Windows the GPU driver provides it).

## Build

Clone with submodules (ggml is vendored at `third_party/ggml`):

```sh
git clone --recursive https://github.com/mudler/parakeet.cpp
cd parakeet.cpp
cmake -B build -DPARAKEET_BUILD_TESTS=ON && cmake --build build -j
```

Use `-DGGML_NATIVE=OFF` for portable or CI builds (it disables host-specific ISA extensions). For the shared library (LocalAI / dlopen):

```sh
cmake -B build-shared -DPARAKEET_SHARED=ON -DPARAKEET_BUILD_CLI=ON
cmake --build build-shared -j
# -> build-shared/libparakeet.so
```

### CMake options

| Option                   | Default | Purpose                                    |
| ------------------------ | ------- | ------------------------------------------ |
| `PARAKEET_BUILD_TESTS`   | OFF     | Compile and register ctest targets         |
| `PARAKEET_BUILD_CLI`     | ON      | Build `parakeet-cli`                       |
| `PARAKEET_SHARED`        | OFF     | Build libparakeet as a shared library      |
| `PARAKEET_VERSION`       | 0.0.1   | Version string returned by `--version`     |
| `PARAKEET_GGML_CUDA`     | OFF     | Forward GGML_CUDA to the submodule         |
| `PARAKEET_GGML_METAL`    | OFF     | Forward GGML_METAL to the submodule        |
| `PARAKEET_GGML_VULKAN`   | OFF     | Forward GGML_VULKAN to the submodule       |
| `PARAKEET_GGML_HIP`      | OFF     | Forward GGML_HIP (ROCm) to the submodule   |
| `PARAKEET_WITH_CED`      | ON      | Sound-event detection through ced.cpp      |
| `PARAKEET_WITH_VOICEDETECT` | ON   | Speaker identification through voice-detect.cpp |

To build for a GPU backend, forward its flag, e.g. Apple Metal:

```sh
cmake -B build -DPARAKEET_GGML_METAL=ON && cmake --build build -j
```

The CLI auto-selects the first GPU device the ggml registry reports (including integrated GPUs such as Ryzen APUs), so no runtime flag is needed. Use `PARAKEET_DEVICE` to override: set it to `cpu` to force CPU, or to a specific device name like `CUDA0` or `Vulkan1` (case-insensitive) to pick that device. Ops the chosen backend has no kernel for run on the CPU automatically, so a model always runs even when one op lacks a GPU kernel. On an Apple M4, Metal is up to about 5x faster than CPU on the larger models; see [Apple Metal](benchmarks/BENCHMARK.md#apple-metal-m4).

---

## Docker

Two prebuilt images are published to GitHub Container Registry on every push to `master`, one per binary:

- `ghcr.io/mudler/parakeet.cpp-cli`: the command-line transcriber.
- `ghcr.io/mudler/parakeet.cpp-server`: the [OpenAI-compatible server](#openai-compatible-server).

Each comes in a CPU and a CUDA variant (the CUDA tag is suffixed `-cuda`), and both are multi-arch (`linux/amd64` and `linux/arm64`), so the right one is pulled for your host automatically. They contain just the binary, so mount a converted `.gguf` model (and, for the cli, your audio) at runtime:

```sh
# CLI, CPU
docker run --rm \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/audio:/audio:ro" \
  ghcr.io/mudler/parakeet.cpp-cli:latest \
  transcribe --model /models/parakeet-tdt_ctc-110m-q5_k.gguf --input /audio/speech.wav --decoder tdt

# CLI, CUDA (needs the nvidia container toolkit on the host)
docker run --rm --gpus all \
  -v "$PWD/models:/models:ro" -v "$PWD/audio:/audio:ro" \
  ghcr.io/mudler/parakeet.cpp-cli:latest-cuda \
  transcribe --model /models/parakeet-tdt_ctc-110m-q5_k.gguf --input /audio/speech.wav --decoder tdt

# Server: binds 0.0.0.0 and exposes 8080. Fetch a model by alias on first run,
# or mount a local .gguf. Add --gpus all with the :latest-cuda tag for GPU.
docker run --rm -p 8080:8080 ghcr.io/mudler/parakeet.cpp-server:latest --model tdt_ctc-110m
```

The CUDA image is built on CUDA 13, so it covers everything from Turing up through Blackwell, including GB10 / Grace-Blackwell (DGX Spark) on arm64.

To build the images yourself, see the build args at the top of the [`Dockerfile`](Dockerfile); the cli is the default target and the server is `--target runtime-server`. The CPU image is the portable `GGML_NATIVE=OFF` build, so it runs on any amd64 or arm64 host.

---

## Python environment setup

You need this once, for model conversion and validation. It's not needed for inference:

```sh
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r scripts/requirements.txt   # nemo_toolkit[asr] + gguf
```

NeMo 2.7.3 is the validated version. The anchor checkpoint `nvidia/parakeet-tdt_ctc-110m` (about 440 MB) is downloaded automatically by NeMo on first use.

---

## Converting a model

Convert a HuggingFace or local `.nemo` checkpoint to GGUF:

```sh
# Default (F32), lossless and largest
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt_ctc-110m \
    --output m.gguf

# F16, about 0.58x the size, WER 0 vs NeMo
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt_ctc-110m --dtype f16 --output m.gguf

# Q8_0, about 0.39x the size, WER 0 vs NeMo
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/parakeet-tdt_ctc-110m --dtype q8_0 --output m.gguf
```

Supported `--dtype`: `f32` (default), `f16`, `q8_0`.

---

## Quantization

The Python `gguf` writer can't produce K-quants (`q4_k`, `q5_k`, `q6_k`), so re-quantize an existing F32 GGUF with the CLI instead:

```sh
parakeet-cli quantize <in.gguf> <out.gguf> <type>
# e.g.
parakeet-cli quantize m.gguf m_q4k.gguf q4_k
parakeet-cli quantize m.gguf m_q6k.gguf q6_k
```

Supported types: `q4_0`, `q5_0`, `q8_0`, `q4_k`, `q5_k`, `q6_k`.

Only the large linear `ggml_mul_mat`-consumed weights (encoder FFN, attention projections, joint enc/pred projections, subsampling output projection) get quantized. The conv, LSTM, featurizer, batch_norm, and bias tensors stay F32. See `docs/quantization.md` for the full policy, allowlist, and measured size and WER per type.

---

## Running inference

```sh
# Default decoder (TDT for hybrid/TDT models, CTC for standalone CTC)
parakeet-cli transcribe --model m.gguf --input audio.wav

# Force a decoder
parakeet-cli transcribe --model m.gguf --input audio.wav --decoder ctc
parakeet-cli transcribe --model m.gguf --input audio.wav --decoder tdt

# Per-word timestamps + confidence: one line per word
#   <start>-<end>  <word>  (<conf>)   (times in seconds)
parakeet-cli transcribe --model m.gguf --input audio.wav --timestamps

# JSON with the flat text plus per-word and per-token timestamps + confidence:
#   {"text":"...","words":[{"w":..,"start":..,"end":..,"conf":..}],
#    "tokens":[{"id":..,"t":..,"conf":..}]}
parakeet-cli transcribe --model m.gguf --input audio.wav --json

# Offline TDT N-best hypotheses as ranked JSON
parakeet-cli transcribe --model m.gguf --input audio.wav --decoder tdt \
  --beam-size 4 --nbest 4

# Read WAV bytes from stdin (useful with ffmpeg/curl pipelines)
ffmpeg -i input.mp3 -f wav - | parakeet-cli transcribe --model m.gguf --input -

# Long audio on Ultra/Redux: cut at VAD pauses, transcribe each piece (offline only).
# Tune with --vad-threshold F (0.5), --vad-min-pause SEC (0.2), --vad-max-seg SEC (30)
parakeet-cli transcribe --model ultra.gguf --input long.wav --vad

# Voice activity detection only, no transcript: speech regions as JSON
#   {"mode":"speech","duration":..,"frame_sec":0.08,"backend":"cpu",
#    "segments":[{"start":..,"end":..}]}   (seconds; models with a VAD head only)
# --mode segments gives the cuts that `transcribe --vad` uses; --probabilities adds p per 80 ms frame.
# Tune with --threshold F (0.5), --min-pause SEC (0.2), --min-speech SEC (0.1), --max-segment SEC (30)
parakeet-cli vad --model ultra.gguf --input audio.wav

# The same with a Silero VAD GGUF (frame_sec 0.032; defaults 250 ms min speech,
# 100 ms min pause, 30 ms pad). Any ASR model can then cut long audio with it:
parakeet-cli vad --model silero-vad-f16.gguf --input audio.wav
parakeet-cli transcribe --model tdt-0.6b-v3.gguf --input long.wav --vad --vad-model silero-vad-f16.gguf

# Print model metadata (arch, dims, mel params, vocab size, TDT durations)
parakeet-cli info m.gguf

# Cache-aware streaming (EOU model parakeet_realtime_eou_120m-v1): feeds the WAV
# in the model's chunk schedule, prints partial text incrementally and
# [EOU @ <t>s] / [EOB @ <t>s] event markers, then the finalized tail. Add
# --timestamps to also print per-word [start-end] (conf) lines as words finalize.
parakeet-cli transcribe --model eou.gguf --input audio.wav --stream
```

Timestamps and confidence match NeMo's `transcribe(timestamps=True)` with the `max_prob` confidence method exactly (word offsets to 0.0 s, per-token and per-word confidence within `5e-6`), for both the TDT and CTC heads. See `docs/parity.md`. Word start and end are in seconds (`frame x hop x subsampling / sample_rate`, which works out to 0.08 s/frame here); confidence is the rescaled softmax probability of the emitted token, aggregated per word with NeMo's `min`.

The optional TDT beam decoder follows NeMo's default sequence-level beam
search and exposes raw/normalized scores plus token frame/duration metadata.
See [`docs/tdt-nbest.md`](docs/tdt-nbest.md).

The `parakeet-cli` binary lands at `build/examples/cli/parakeet-cli`.

---

## OpenAI-compatible server

`parakeet-server` is a small HTTP server that speaks the OpenAI transcription
API, so any OpenAI client works by pointing its `base_url` at it. It is built by
default (`PARAKEET_BUILD_SERVER=ON`) and lands at `build/examples/server/parakeet-server`.

```sh
# Serve a model. --model takes a local .gguf, an http(s) URL, a <name>.gguf in
# mudler/parakeet-cpp-gguf, or an alias (downloaded and cached on first run).
parakeet-server --model tdt_ctc-110m --port 8080

# Transcribe over HTTP
curl -F file=@audio.wav -F response_format=verbose_json \
  http://localhost:8080/v1/audio/transcriptions
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
with open("audio.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="parakeet", file=f).text)
```

It supports `response_format` `json` / `text` / `verbose_json` and
`timestamp_granularities[]=word`. This is a single-model, one-request-at-a-time
example that accepts WAV uploads only; see [`examples/server/README.md`](examples/server/README.md)
for the full list of options and known simplifications. **For a production
deployment, use [LocalAI](https://localai.io)**, which embeds parakeet.cpp as a
backend and adds a model gallery, concurrency, multi-model serving, the full
OpenAI API surface, auth, and metrics.

---

## Batching

Single-clip transcription is the default and needs no flags: every `transcribe` call runs one clip at a time, byte-for-byte identical to before. Batching is an opt-in path for decoding several clips together, which matters when you serve many concurrent requests on a GPU.

The win is on the **decode** side. A transducer (TDT/RNN-T) decodes autoregressively with tiny per-step prediction-LSTM and joint GEMMs; one clip launches hundreds of these matvec-sized kernels and leaves the GPU mostly idle between launches. Decoding N clips together coalesces each step into one batched GEMM, so the device stays busy. On the NVIDIA GB10 this reaches about **10-12x** at batch size 16 (CPU about 3-5x); the encoder is already compute-bound, so batching it gives no throughput win. CTC has no autoregressive decode, so batching does not apply to standalone CTC models. On CPU the batched decode step is bit-identical to decoding each clip alone: its matmuls call the same dot-product kernel as a single column, so logits, token ids, frames and confidences are equal (`tests/test_exact_batch.cpp` checks this for F32, F16 and Q8_0 decoder weights). On GPU backends the batched matmul is the ordinary ggml kernel, so logits agree to about 1e-4 rather than exactly. The batched encoder is separate: its output is close to the single-clip encoder output, not equal, so a whole batched transcript can still differ from a single-clip one, and the tests compare with a tolerance there. Full numbers and per-model tables are in [`benchmarks/BENCHMARK.md`](benchmarks/BENCHMARK.md#batched-decode-throughput).

Measure it yourself:

```bash
# Decode-only: serial vs batched decode of one clip replicated B times (the win in isolation).
parakeet-cli bench-decode --model <model.gguf> --audio <wav> [--batch-sizes 1,4,8,16] [--threads N] [--reps R] [--json <out>]

# Full transcribe (encoder + decode) over a manifest at several batch sizes.
parakeet-cli bench-batch --model <model.gguf> --manifest <file> [--decoder ctc|tdt] [--threads N] [--batch-sizes 1,4,8] [--json <out>]
```

To batch from code, use the batched entry points (single-clip B=1 is just N=1):

- C++ (`src/model.hpp`): `Model::transcribe_16k_batch(pcms16k, decoder)` and `transcribe_16k_batch_with_timestamps(...)` take N clips of 16 kHz mono float PCM and return N results.
- C-API (`include/parakeet_capi.h`): `parakeet_capi_transcribe_pcm_batch(...)` (N transcripts) and `parakeet_capi_transcribe_pcm_batch_json(...)` (one JSON array of N `{text,words,tokens}` objects). These are what LocalAI's `parakeet-cpp` backend calls to coalesce concurrent requests; it leaves batching off by default and exposes a `batch_max_size` option to opt in.

---

## Concurrent requests

One loaded model runs one request at a time by default. To serve several requests in parallel, give the model a pool of CPU backends: `parakeet_capi_set_concurrency(ctx, backends, threads_each)`, `pk::Model::set_concurrency`, or `--concurrency K` on `parakeet-server` and `parakeet-cli bench`. Results are identical to the single-backend run, aggregate throughput can go up or down depending on model size and core count (about 1.2x to 1.3x on a 110M model with 8 cores, but slower than one backend on 0.6B models with 8 threads; all measured on loaded machines, see [`docs/concurrency.md`](docs/concurrency.md)), and each request gets somewhat slower because it has fewer threads. Try it only when `backends x threads_each` fits the physical cores, measure before you enable it, and expect extra memory per backend (`Model::pool_working_set_bytes()`). The default is one backend and behaves as before.

---

## Sound events

parakeet.cpp can also tag everyday sounds (dog bark, glass breaking, applause,
alarms, music, and the rest of the 527-class AudioSet ontology) with
[CED](https://github.com/RicherMans/CED), through the
[ced.cpp](https://github.com/localai-org/ced.cpp) submodule (`PARAKEET_WITH_CED`,
on by default). `parakeet-cli scene` combines it with ASR and diarization into
one time-ordered feed:

```sh
parakeet-cli scene --model asr.gguf --diar diar.gguf --sound ced-base-q8_0.gguf \
  --latency low --input audio.wav
[00:00.4 - 00:03.2]  Speaker 0: mister Quilter is the apostle of the middle classes, and
[00:24.0 - 00:30.0]  (Chicken, rooster 0.86)
```

See [`docs/sound.md`](docs/sound.md) for the CED GGUFs, the sound and scene
stream C-API (ABI v8), and the `--sound-model` server option.

### Naming speakers

With a voice-detect.cpp speaker encoder (`PARAKEET_WITH_VOICEDETECT`, on by
default) the scene stream can say who is talking instead of `Speaker 0`.
Enroll each person from a short clip with `parakeet-cli enroll`, then pass
`--speakers <speaker.gguf> --registry <file>` to `scene`. Only one two-voice
fixture has been measured so far. See [`docs/speaker.md`](docs/speaker.md) for
the models, the commands, the C-API (ABI v9 and v10) and what is still untested.

---

## C-API (`libparakeet.so`)

`include/parakeet_capi.h` defines a flat, exception-free C-API meant for `dlopen` / FFI / LocalAI integration. Build the shared library with `-DPARAKEET_SHARED=ON`:

```c
#include "parakeet_capi.h"

parakeet_ctx *ctx = parakeet_capi_load("model.gguf");  // load ONCE
if (!ctx) { fprintf(stderr, "%s\n", parakeet_capi_last_error(ctx)); return 1; }

char *text = parakeet_capi_transcribe_path(ctx, "audio.wav", 0 /*default*/);
if (text) { printf("%s\n", text); parakeet_capi_free_string(text); }

parakeet_capi_free(ctx);
```

In-memory PCM:
```c
char *text = parakeet_capi_transcribe_pcm(ctx, samples, n_samples,
                                          sample_rate, 0 /*default*/);
```

Timestamps and confidence as JSON (matches NeMo `timestamps=True` + `max_prob`):
```c
char *json = parakeet_capi_transcribe_path_json(ctx, "audio.wav", 0 /*default*/);
// {"text":"...",
//  "frame_sec":0.080000,
//  "words":[{"w":"Well,","start":0.480,"end":0.640,"conf":0.7859}, ...],
//  "tokens":[{"id":639,"t":0.480,"conf":0.9969}, ...]}
if (json) { printf("%s\n", json); parakeet_capi_free_string(json); }
```
`start`/`end`/`t` are in seconds; `conf` is the rescaled softmax probability of the emitted token in `(0,1]` (a word's `conf` is the `min` over its tokens). `frame_sec` is the encoder frame stride in seconds (`hop x subsampling / sample_rate`); multiply a frame-unit segment gap threshold (NeMo's `segment_gap_threshold`) by it to get the seconds gap between words when forming segments.

### Streaming (cache-aware EOU model)

For `parakeet_realtime_eou_120m-v1`, a streaming session decodes 16 kHz mono f32 PCM as it arrives, returning newly-finalized text and signalling EOU/EOB events:

```c
parakeet_stream *s = parakeet_capi_stream_begin(ctx);
int eou = 0;
char *t = parakeet_capi_stream_feed(s, pcm, n_samples, &eou); // "" if none yet
if (t) { printf("%s", t); parakeet_capi_free_string(t); }
if (eou) printf(" [EOU]");
// ...feed more chunks...
char *tail = parakeet_capi_stream_finalize(s);                // flush the tail
if (tail) { printf("%s\n", tail); parakeet_capi_free_string(tail); }
parakeet_capi_stream_free(s);
```

`<EOU>` (end-of-utterance) and `<EOB>` (backchannel) are stripped from the text and surfaced via `*eou_out` (the CLI `--stream` prints them as `[EOU @ <t>s]` markers). The streaming transcript matches NeMo's cache-aware streaming exactly, and `finalize` flushes the end-of-stream tail without fabricating an `<EOU>` that NeMo would not emit.

The LocalAI backend (in the LocalAI repo) dlopens `libparakeet.so` and uses these symbols directly: the offline `parakeet_capi_transcribe_*` / `parakeet_capi_transcribe_path_json` and the streaming `parakeet_capi_stream_*`. See `include/parakeet_capi.h` for the full API. The C++ streaming session (`pk::StreamingSession`) also exposes per-word timestamps and confidence as words finalize, via `drain_words()` alongside the EOU events, which the CLI `--stream --timestamps` path prints.

---

## Model coverage

See `docs/parity.md` for the full coverage matrix. In short:

| Family | Representative checkpoints | Heads | WER vs NeMo |
| --- | --- | --- | --- |
| Hybrid TDT+CTC | `parakeet-tdt_ctc-110m`, `parakeet-tdt_ctc-1.1b` | TDT + CTC | 0.0 |
| TDT (hybrid) | `parakeet-tdt-0.6b-v2`, `parakeet-tdt-0.6b-v3` (multilingual) | TDT | 0.0 |
| Pure TDT | `parakeet-tdt-1.1b` | TDT | 0.0 |
| CTC | `parakeet-ctc-0.6b`, `parakeet-ctc-1.1b` | CTC | 0.0 |
| RNNT | `parakeet-rnnt-0.6b`, `parakeet-rnnt-1.1b` | RNNT | 0.0 |

All 10 published offline checkpoints are validated at WER 0 vs NeMo 2.7.3. Sizes: 110M (512/17 layers), 0.6B (1024/24), 1.1B (1024/42).

Cache-aware streaming and EOU (`parakeet_realtime_eou_120m-v1`) is implemented too: `layer_norm` plus causal conv, causal subsampling, chunked-limited attention, per-layer conv/attention caches, carried RNN-T decoder state, and `<EOU>`/`<EOB>` events. The streaming transcript matches NeMo's cache-aware streaming byte for byte. See `docs/parity.md` (the Streaming + EOU section).

---

## Running tests

Model-independent (run anywhere):

```sh
ctest --test-dir build --output-on-failure -LE model
```

Model-dependent (need venv + checkpoint):

```sh
export PARAKEET_TEST_GGUF=/tmp/pk110m.gguf
export PARAKEET_TEST_BASELINE=/tmp/baseline.gguf
export PARAKEET_TEST_BASELINE_SPEECH=/tmp/baseline_speech.gguf
ctest --test-dir build --output-on-failure
```

Tests labelled `model` return exit code 77 (ctest SKIP) when their required env vars are absent, so they never break a CI environment that has no model.

---

## Roadmap / TODO

- **Tune the GPU encoder kernels.** On the GB10 GPU, parakeet.cpp is faster than NeMo on all 10 models (median 1.25x, up to 4.3x), but the gains are smallest on the pure-encoder CTC models (around 1.2x), because ggml's generic CUDA conv/attention kernels still trail NeMo's tuned cuDNN. Closing that gap (better conv1d and flash-attention paths for the FastConformer encoder) is the main remaining GPU headroom. The log-mel already runs on the backend (`GpuMel`); the CPU path is unaffected.

---

## Why parakeet.cpp

NeMo is a great training framework, but running Parakeet just for inference drags in a heavy Python/PyTorch stack. parakeet.cpp is a from-scratch C++17/ggml port focused purely on inference:

- **No Python at inference.** A single `libparakeet.so` (or static lib) behind a flat C API (`include/parakeet_capi.h`), easy to embed from C, C++, Go, or Rust.
- **Faster than NeMo** on CPU and GPU (see [Performance](#performance)), with byte-identical output.
- **Small and portable.** GGUF models with f16 / q8_0 / K-quant variants, running on CPU and any ggml GPU backend (CUDA, Metal, Vulkan, HIP).
- **Full family coverage.** CTC, RNNT, TDT, hybrid TDT-CTC, multilingual, and cache-aware streaming with EOU, all validated at WER 0 vs NeMo.

---

## Community projects

Built on parakeet.cpp by the community (not maintained or tested by the core team):

- [**parakeet-ios-demo**](https://github.com/Kashif-E/parakeet-ios-demo) — live,
  on-device streaming speech-to-text on iOS (SwiftUI) over the streaming C-API,
  with a side-by-side compare against Apple's SpeechTranscriber and Moonshine.
  By [@Kashif-E](https://github.com/Kashif-E).

---

## Citation

If you use parakeet.cpp, please cite this repository and the original models:

```bibtex
@software{parakeet_cpp,
  title  = {parakeet.cpp: a C++/ggml inference engine for NVIDIA Parakeet ASR},
  author = {Di Giacinto, Ettore and Palethorpe, Richard},
  url    = {https://github.com/mudler/parakeet.cpp},
  year   = {2026}
}
```

The Parakeet models are by NVIDIA NeMo ([NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo)).
Parakeet Ultra and Redux are by [Moondream](https://huggingface.co/moondream), derived from NVIDIA's parakeet-tdt-0.6b-v3.

## Author

Ettore Di Giacinto ([@mudler](https://github.com/mudler)).

## License

parakeet.cpp is released under the [MIT License](LICENSE). The model weights are governed by the licenses of the original models, so check each model card on HuggingFace. The NVIDIA Parakeet models are mostly CC-BY-4.0 (nemotron-3.5-asr-streaming is OpenMDW-1.1). Moondream's [parakeet-ultra](https://huggingface.co/moondream/parakeet-ultra) and [parakeet-redux](https://huggingface.co/moondream/parakeet-redux) are CC-BY-4.0 too: credit Moondream and NVIDIA ([parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)), link the [license](https://creativecommons.org/licenses/by/4.0/), and note that GGUF files made here are converted (and quantized, or dequantized for Redux) copies, not retrained models.
