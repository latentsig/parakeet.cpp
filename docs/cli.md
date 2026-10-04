# Command-line reference

`parakeet-cli` lands at `build/examples/cli/parakeet-cli`. The README has a short cheat sheet; this page has the full set of examples.

## Transcribe, VAD, info and streaming

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

# Only the VAD head is needed? Use a 6 to 10 MB slice instead of the full model
# (it cannot transcribe). Files and checksums: ./vad.md.
curl -LO https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/redux-vad.gguf
parakeet-cli vad --model redux-vad.gguf --input audio.wav

# The same with a Silero VAD GGUF (frame_sec 0.032; defaults 250 ms min speech,
# 100 ms min pause, 30 ms pad). Any ASR model can then cut long audio with it.
# Download it from the collection repo (F16 is 1.3 MB, F32 is 2.2 MB). To make the
# file yourself, see scripts/convert_silero_vad_to_gguf.py, ./vad.md and ./conversion.md.
curl -LO https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/silero-vad-f16.gguf
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

Timestamps and confidence match NeMo's `transcribe(timestamps=True)` with the `max_prob` confidence method exactly (word offsets to 0.0 s, per-token and per-word confidence within `5e-6`), for both the TDT and CTC heads. See `./parity.md`. Word start and end are in seconds (`frame x hop x subsampling / sample_rate`, which works out to 0.08 s/frame here); confidence is the rescaled softmax probability of the emitted token, aggregated per word with NeMo's `min`.

The optional TDT beam decoder follows NeMo's default sequence-level beam
search and exposes raw/normalized scores plus token frame/duration metadata.
See [`./tdt-nbest.md`](./tdt-nbest.md).

The `parakeet-cli` binary lands at `build/examples/cli/parakeet-cli`.

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
example that accepts WAV uploads only; see [`examples/server/README.md`](../examples/server/README.md)
for the full list of options and known simplifications. **For a production
deployment, use [LocalAI](https://localai.io)**, which embeds parakeet.cpp as a
backend and adds a model gallery, concurrency, multi-model serving, the full
OpenAI API surface, auth, and metrics.

## Quantize

The Python `gguf` writer can't produce K-quants (`q4_k`, `q5_k`, `q6_k`), so re-quantize an existing F32 GGUF with the CLI instead:

```sh
parakeet-cli quantize <in.gguf> <out.gguf> <type>
# e.g.
parakeet-cli quantize m.gguf m_q4k.gguf q4_k
parakeet-cli quantize m.gguf m_q6k.gguf q6_k
```

Supported types: `q4_0`, `q5_0`, `q8_0`, `q4_k`, `q5_k`, `q6_k`.

Only the large linear `ggml_mul_mat`-consumed weights (encoder FFN, attention projections, joint enc/pred projections, subsampling output projection) get quantized. The conv, LSTM, featurizer, batch_norm, and bias tensors stay F32. See [quantization.md](quantization.md) for the full policy, allowlist, and measured size and WER per type.
