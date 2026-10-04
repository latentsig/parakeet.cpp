# Batched decode

Single-clip transcription is the default and needs no flags: every `transcribe` call runs one clip at a time, byte-for-byte identical to before. Batching is an opt-in path for decoding several clips together, which matters when you serve many concurrent requests on a GPU.

The win is on the **decode** side. A transducer (TDT/RNN-T) decodes autoregressively with tiny per-step prediction-LSTM and joint GEMMs; one clip launches hundreds of these matvec-sized kernels and leaves the GPU mostly idle between launches. Decoding N clips together coalesces each step into one batched GEMM, so the device stays busy. On the NVIDIA GB10 this reaches about **10-12x** at batch size 16 (CPU about 3-5x); the encoder is already compute-bound, so batching it gives no throughput win. CTC has no autoregressive decode, so batching does not apply to standalone CTC models. On CPU the batched decode step is bit-identical to decoding each clip alone: its matmuls call the same dot-product kernel as a single column, so logits, token ids, frames and confidences are equal (`tests/test_exact_batch.cpp` checks this for F32, F16 and Q8_0 decoder weights). On GPU backends the batched matmul is the ordinary ggml kernel, so logits agree to about 1e-4 rather than exactly. The batched encoder is separate: its output is close to the single-clip encoder output, not equal, so a whole batched transcript can still differ from a single-clip one, and the tests compare with a tolerance there. Full numbers and per-model tables are in [`../benchmarks/BENCHMARK.md`](../benchmarks/BENCHMARK.md#batched-decode-throughput).

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
