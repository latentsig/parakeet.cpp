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

Machine: AMD Ryzen 9 9950X3D (16 cores, AVX-512 VNNI), CPU backend, 2026-09-29.
The end-to-end and LibriSpeech numbers were taken at commit c8499e6 (kernel and
encoder code); the microbench at f7c9e8c, which only adds the ggml rows to
`bench_ternary`. Ultra has the same architecture and shapes as v3, so
`ultra-q8_0` (converted from the HF weights with `--dtype q8_0`) stands in for a
Q8_0 v3. The dequantized Redux was also converted to Q8_0 the same way.

### Long clip, one utterance

`transcribe --decoder tdt` on `benchmarks/audio/diverse/i_have_a_dream.wav`
(180 s), whole process wall time including model load, median of 5 runs, 8
threads on cores 0-7, serialized with `flock`. RTF is 180 s divided by the median.

| Model | Form | Size on disk | Median | RTF |
|---|---|---:|---:|---:|
| parakeet-redux | packed ternary, vnni kernel | 213.3 MB | 10.32 s | 17.4 |
| parakeet-redux | packed ternary, avx2 kernel | 213.3 MB | 10.66 s | 16.9 |
| parakeet-redux | packed ternary, scalar kernel (1 run) | 213.3 MB | 116.39 s | 1.5 |
| parakeet-redux | dequantized Q8_0 | 941.5 MB | 9.06 s | 19.9 |
| parakeet-redux | dequantized F16 | 1441.9 MB | 8.98 s | 20.0 |
| parakeet-ultra | Q8_0 | 941.5 MB | 8.97 s | 20.1 |
| parakeet-ultra | F16 | 1441.9 MB | 8.90 s | 20.2 |
| parakeet-tdt-0.6b-v3 | F16 | 1441.0 MB | 31.25 s | 5.8 |

With 16 threads (cores 0-15) the F16 and ternary rows move to: ternary vnni
9.73 s, ternary avx2 9.88 s, Redux F16 8.30 s, Ultra F16 8.38 s, v3 F16 35.13 s.

### Per utterance, LibriSpeech

`parakeet-cli bench --manifest benchmarks/librispeech_manifest.tsv --decoder tdt
--threads 8` under `taskset -c 0-7`: 100 utterances, 901.1 s of audio, one
utterance at a time. RTF is total audio over the summed per-utterance processing
time, median of 3 full passes (the first pass after a model is read from disk
was the slowest for every model). WER is against the manifest text after
lowercasing and stripping punctuation; it is from a quick script, not
`validate_vs_nemo.py`, and is identical across passes.

| Model | Form | RTF pass 1 / 2 / 3 | Median RTF | WER |
|---|---|---|---:|---:|
| parakeet-ultra | Q8_0 | 38.1 / 44.3 / 41.8 | 41.8 | 1.71% |
| parakeet-redux | dequantized F16 | 36.4 / 45.1 / 44.9 | 44.9 | 1.92% |
| parakeet-redux | dequantized Q8_0 | 35.1 / 43.2 / 40.4 | 40.4 | 1.96% |
| parakeet-redux | packed ternary, vnni | 32.1 / 39.8 / 39.3 | 39.3 | 1.96% |
| parakeet-redux | packed ternary, avx2 | 36.2 / 37.7 / 38.7 | 37.7 | 1.96% |

### Single-thread kernel ceiling

`build/tests/bench_ternary N K T 10`, pinned to one core. The ggml rows are
`ggml_mul_mat` with an F32 activation matrix on one thread, so they include
ggml's own activation quantization; the ternary rows time only the matmul
kernel and leave out `ternary_quant_rows`.

| N x K, T | scalar | ternary avx2 | ternary vnni | ggml Q8_0 | ggml F16 |
|---|---:|---:|---:|---:|---:|
| 4096 x 1024, T=200 | 1.97 | 78.14 | 81.91 | 87.23 | 120.46 |
| 1024 x 4096, T=200 | 1.96 | 78.63 | 87.01 | 87.91 | 120.76 |
| 1024 x 4096, T=1000 | 1.97 | 66.79 | 82.57 | 89.37 | 125.86 |

All numbers are GMAC/s (raw output in the task 8 report; a second run of the
same command gave values within a few percent of these). Because the ternary
rows exclude the int8 activation quantization and the ggml rows include their
own, the true gap of ternary to ggml is somewhat larger than this table shows.

### What the data say

- The ternary form is 6.8 times smaller than F16 and 4.4 times smaller than
  Q8_0, but on this machine it is not faster. The ternary vnni kernel is slower
  than ggml's Q8_0 kernel in all three shapes (by 6, 1 and 8 percent) and 28 to
  34 percent slower than ggml's F16 kernel.
- End to end, packed ternary is slower than both Q8_0 and F16 of the same
  model: 10.32 s against 9.06 s and 8.98 s on the long clip, and a median RTF
  of 39.3 against 40.4 and 44.9 on LibriSpeech. It is also slower than the
  ternary-free Ultra Q8_0 (41.8 on LibriSpeech). The gap is small and the
  LibriSpeech passes are noisy (the Redux F16 passes went from 36.4 to 45.1,
  about 24 percent), so read it as "no speed win", not as a precise ratio.
- The avx2 kernel is close to the vnni kernel here (78.14 against 81.91 GMAC/s at
  N=4096, K=1024, T=200), so the VNNI instruction adds little. The cause was not profiled.
- The gain of the ternary form on this machine is size (and memory traffic),
  not speed. The speed comparison on other CPUs (NEON, fewer cores, less
  bandwidth) has not been made.
- The v3 F16 row is 3.5 times slower than Ultra and Redux F16 with identical
  shapes on the long clip. That gap was not investigated.
- The scalar kernel is a reference only.

Transcripts on `tests/fixtures/speech.wav` are identical across the scalar,
avx2 and vnni kernels and the dequantized F16 and Q8_0 Redux, and equal the
reference transcript in `AGENTS.md`.

## Tests

```
ctest --test-dir build -R ternary --output-on-failure        # test_ternary (kernels vs scalar), no model needed

PARAKEET_TEST_GGUF_REDUX_KEEP=<redux --ternary keep gguf> \
PARAKEET_TEST_GGUF_REDUX_DEQ=<redux --ternary dequant gguf> \
    ctest --test-dir build -R test_ternary_model --output-on-failure

build/tests/bench_ternary [N K T reps]                        # per-kernel single thread throughput, plus ggml Q8_0 and F16 mul_mat
```
