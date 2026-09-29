# Ternary encoders (moondream/parakeet-redux)

## What Redux is

`moondream/parakeet-redux` is a Parakeet TDT 0.6B v3 derivative whose encoder
linear layers are ternary: every weight is -1, 0 or +1 times a per-group scale.
Its sibling `moondream/parakeet-ultra` has the same shapes but ordinary F16
weights, so it needs no special support and loads like any other v3 GGUF.

In the encoder the ternary layers are the two FFN pairs (`linear1`, `linear2`
in each of `feed_forward1` and `feed_forward2`), the attention projections
(`linear_q/k/v/out/pos`) and the two pointwise convolutions, which is 11
layers per block and 264 in the 24 block encoder. Norms, biases, the
depthwise conv, the subsampling stack, the prediction net and the joint stay
in ordinary float tensors.

## GGUF form

Produce it with `scripts/convert_hf_parakeet_to_gguf.py --ternary keep` (see
`docs/conversion.md`). For each ternary linear `<base>` the file holds:

| Tensor | Type | Content |
|---|---|---|
| `<base>.qweight` | I8 | the upstream bytes, N rows of ceil(K/5) bytes (5 base-3 digits per byte) |
| `<base>.scales` | F16 | one scale per row and 128 input columns, shape N x K/128 |

and no `<base>.weight`. Two KVs mark the file: `parakeet.ternary.present`
(bool) and `parakeet.ternary.group_size` (u32, 128). The loader reads them into
`ParakeetConfig::ternary`. A packed Redux GGUF is 213.3 MB on disk, against
1441.0 MB for the v3 F16 file.

## Runtime path

1. At load time `ternary_prepare` repacks every `qweight` once into a
   kernel layout: per row, 32 bytes per 128 columns, with the 2-bit codes of
   elements j, j+32, j+64 and j+96 in one byte. The packed bytes stay in the
   GGUF as they are; nothing is dequantized per call.
2. At run time the activations of each ternary linear are quantized to int8
   per token (one float scale and one int32 sum per 128-column group).
3. The layer is a pair of ggml custom ops: one quantizes the activations, one
   runs the integer matmul and applies the scales. The dot product uses
   `code * int8` sums with the constant offset removed through the group sums.
4. The kernel is picked once per process: AVX-512 VNNI, then NEON dot product,
   then AVX2, then a scalar reference. All kernels are bit-identical to the
   scalar reference. `PARAKEET_TERNARY_KERNEL=scalar|avx2|vnni|neon` forces one
   (an unavailable name falls back to automatic selection with a log line).

## Limits

- CPU only. Loading a packed GGUF with a GPU backend active fails with a
  message; re-convert with `--ternary dequant` to use a GPU.
- No cache-aware streaming. `StreamingEncoder` rejects packed GGUFs; use a
  `--ternary dequant` file for streaming.
- `parakeet-cli quantize` never touches ternary tensors. Running it on a packed
  file copies `.qweight` (I8) and `.scales` (F16) verbatim.

## Measured speed

`transcribe --decoder tdt` on `benchmarks/audio/diverse/i_have_a_dream.wav`
(180 s, 16 kHz mono), whole process wall time including model load, median of
5 runs, serialized with `flock`. Machine: AMD Ryzen 9 9950X3D (16 cores, AVX-512
VNNI), CPU backend, commit c8499e6 plus these docs, 2026-09-29. RTF is 180 s
divided by the median.

| Model | Form | Size on disk | 8 threads (cores 0-7) | RTF | 16 threads (cores 0-15) | RTF |
|---|---|---:|---:|---:|---:|---:|
| parakeet-redux | packed ternary, vnni kernel | 213.3 MB | 10.32 s | 17.4 | 9.73 s | 18.5 |
| parakeet-redux | packed ternary, avx2 kernel | 213.3 MB | 10.66 s | 16.9 | 9.88 s | 18.2 |
| parakeet-redux | packed ternary, scalar kernel (1 run) | 213.3 MB | 116.39 s | 1.5 | not run | |
| parakeet-redux | dequantized F16 | 1441.9 MB | 8.98 s | 20.0 | 8.30 s | 21.7 |
| parakeet-ultra | F16 | 1441.9 MB | 8.90 s | 20.2 | 8.38 s | 21.5 |
| parakeet-tdt-0.6b-v3 | F16 | 1441.0 MB | 31.25 s | 5.8 | 35.13 s | 5.1 |

Notes on reading the table:

- The ternary form is 6.8 times smaller than F16, but on this machine it is
  not faster than the dequantized F16 Redux (about 15 percent slower at 8
  threads). The ggml F16 matmul is already fast on AVX-512 here. The win of the
  ternary form is size and memory traffic, and it should matter more on
  machines with fewer cores or less memory bandwidth (NEON is untested here).
- The v3 F16 row is slower than the Ultra and Redux rows although the shapes
  are identical. That gap was not investigated in this task. No Q8_0 v3 row is
  included: `parakeet-cli quantize` only quantizes F32 inputs, and the
  available v3 GGUF is F16.
- Single-thread kernel throughput from `build/tests/bench_ternary` (N=4096,
  K=1024, T=200): scalar 1.68 GMAC/s, avx2 78.45 GMAC/s, vnni 80.81 GMAC/s.
- Raw per-run times are in the task report; the first run after a model is
  first read from disk can be several seconds slower than the rest.

Transcripts on `tests/fixtures/speech.wav` are identical across the scalar,
avx2 and vnni kernels and the dequantized F16 Redux, and equal the reference
transcript in `AGENTS.md`.

## Tests

```
ctest --test-dir build -R ternary --output-on-failure        # test_ternary (kernels vs scalar), no model needed

PARAKEET_TEST_GGUF_REDUX_KEEP=<redux --ternary keep gguf> \
PARAKEET_TEST_GGUF_REDUX_DEQ=<redux --ternary dequant gguf> \
    ctest --test-dir build -R test_ternary_model --output-on-failure

build/tests/bench_ternary [N K T reps]                        # per-kernel single thread throughput
```
