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
   kernel layout with 2 bits per weight (the same size as the upstream
   bytes). Rows are grouped in blocks of 16, padded with zeros at the end.
   For each block and each run of 16 columns there are 64 bytes: byte
   4*i + j holds the codes of row i at columns 4p + j of the run, for
   p = 0..3, in its four bit pairs. The kernels keep one output row per
   vector lane, so a group sum for 16 rows comes out in one vector with no
   horizontal reduction. The exact layout is in `TernaryWeight` in
   `src/ternary.hpp`. Nothing is dequantized per call.
2. At run time the activations of each ternary linear are quantized to int8
   per token (one float scale and one int32 sum per 128-column group). The
   scalar version defines the bytes; the AVX-512 and AVX2 versions write the
   same bytes.
3. The layer is a pair of ggml custom ops: one quantizes the activations, one
   runs the integer matmul and applies the scales. The dot product uses
   `code * int8` sums with the constant offset removed through the group sums.
4. The kernel is picked once per process: AVX-512 VNNI, then NEON dot product,
   then AVX2, then a scalar reference. All kernels are bit-identical to the
   scalar reference. `PARAKEET_TERNARY_KERNEL=scalar|avx2|vnni|neon` forces one
   (an unavailable name falls back to automatic selection with a log line).
   The activation quantizer is picked by CPU features only.

## Limits

- CPU only. Loading a packed GGUF with a GPU backend active fails with a
  message; re-convert with `--ternary dequant` to use a GPU.
- No cache-aware streaming. `StreamingEncoder` rejects packed GGUFs; use a
  `--ternary dequant` file for streaming.
- `parakeet-cli quantize` never touches ternary tensors. Running it on a packed
  file copies `.qweight` (I8) and `.scales` (F16) verbatim.

## Measured speed

Machine: AMD Ryzen 9 9950X3D (16 cores, Zen 5, AVX-512 VNNI), CPU backend,
2026-09-29. The end-to-end numbers were taken at commit f228376, the microbench
at 8b66c63 (which only changes the compile target of the activation
quantizer). Ultra has the same architecture and shapes as v3, so `ultra-q8_0`
(converted from the HF weights with `--dtype q8_0`) stands in for a Q8_0 v3.
The dequantized Redux was also converted to Q8_0 the same way. Every timed
command was serialized with `flock /tmp/pk-bench.lock`.

### Per utterance, LibriSpeech

```
taskset -c 0-7 build/examples/cli/parakeet-cli bench --model <gguf> \
    --manifest benchmarks/librispeech_manifest.tsv --decoder tdt --threads 8 --json <out.json>
```

100 utterances, 901.1 s of audio, one utterance at a time. RTF is total audio
over the summed per-utterance `proc_ms` of the JSON, median of 3 full passes.
WER is against the manifest text after lowercasing and replacing every
character other than a-z and the apostrophe with a space; it is from a quick
script, not `validate_vs_nemo.py`, and is identical across passes. The kernel
rows set `PARAKEET_TERNARY_KERNEL`; "automatic" leaves it unset, which picks
vnni here.

| Model | Form | RTF pass 1 / 2 / 3 | Median RTF | WER |
|---|---|---|---:|---:|
| parakeet-redux | packed ternary, automatic (vnni) | 74.8 / 75.6 / 78.0 | 75.6 | 1.96% |
| parakeet-redux | packed ternary, avx2 | 59.6 / 59.0 / 58.8 | 59.0 | 1.96% |
| parakeet-redux | dequantized F16 | 46.6 / 46.1 / 45.6 | 46.1 | 1.92% |
| parakeet-redux | dequantized Q8_0 | 42.7 / 42.3 / 42.7 | 42.7 | 1.96% |
| parakeet-ultra | Q8_0 | 42.5 / 43.1 / 44.2 | 43.1 | 1.71% |
| parakeet-redux | packed ternary, previous kernel (b4164da) | 40.3 / 40.2 / 40.1 | 40.2 | 1.96% |

The last row is the kernel before the current one (one row per call, 256-bit
vectors, scalar activation quantization), built from commit b4164da and run in
the same session. The transcripts of the old and the new kernel are identical
for all 100 utterances, as expected from bit-identical kernels.

### Long clip, one utterance

`transcribe --decoder tdt` on `benchmarks/audio/diverse/i_have_a_dream.wav`
(180 s), whole process wall time including model load, median of 5 runs:

```
taskset -c 0-7 build/examples/cli/parakeet-cli transcribe --model <gguf> \
    --input benchmarks/audio/diverse/i_have_a_dream.wav --decoder tdt --threads 8
```

RTF is 180 s divided by the median.

| Model | Form | Size on disk | 8 threads (cores 0-7) | RTF | 16 threads (cores 0-15) |
|---|---|---:|---:|---:|---:|
| parakeet-redux | packed ternary, vnni kernel | 213.3 MB | 7.93 s | 22.7 | 7.45 s |
| parakeet-redux | packed ternary, avx2 kernel | 213.3 MB | 8.52 s | 21.1 | 8.01 s |
| parakeet-redux | dequantized F16 | 1441.9 MB | 8.73 s | 20.6 | 7.75 s |
| parakeet-redux | dequantized Q8_0 | 941.5 MB | 9.04 s | 19.9 | 8.11 s |
| parakeet-ultra | Q8_0 | 941.5 MB | 9.04 s | 19.9 | 8.29 s |
| parakeet-ultra | F16 | 1441.9 MB | 8.63 s | 20.9 | 7.83 s |
| parakeet-tdt-0.6b-v3 | F16 | 1441.0 MB | 32.79 s | 5.5 | 29.29 s |

The scalar kernel is a reference only (116.39 s for this clip in one run at
commit c8499e6, with an earlier and faster form of the reference).

### Single-thread kernel throughput

```
taskset -c 2 build/tests/bench_ternary 4096 1024 200 10
taskset -c 2 build/tests/bench_ternary 1024 4096 200 10
taskset -c 2 build/tests/bench_ternary 1024 4096 1000 10
```

The ggml rows are `ggml_mul_mat` with an F32 activation matrix on one thread,
so they include ggml's own activation quantization; the ternary rows time only
the matmul kernel. `bench_ternary` times `ternary_quant_rows` on its own line.

| N x K, T | scalar | ternary avx2 | ternary vnni | ggml Q8_0 | ggml F16 | ternary quant |
|---|---:|---:|---:|---:|---:|---:|
| 4096 x 1024, T=200 | 1.04 | 184.12 | 546.86 | 87.51 | 121.21 | 0.014 ms |
| 1024 x 4096, T=200 | 1.05 | 184.38 | 553.21 | 91.28 | 129.36 | 0.060 ms |
| 1024 x 4096, T=1000 | 1.05 | 183.27 | 492.23 | 89.85 | 125.12 | 0.408 ms |

Kernel columns are GMAC/s. For comparison, the previous kernel (b4164da)
measured 81.91, 87.01 and 82.57 GMAC/s (vnni) and 78.14, 78.63 and 66.79
(avx2) with the same commands. The quantization of the activations took 1.405 ms
for K=4096, T=200 before it was vectorized, about as long as the new matmul.
Repeated runs of the same commands moved the vnni numbers by up to about 10
percent (the T=1000 shape measured between 492 and 542 over five runs).

### What the data say

- The vnni kernel does 490 to 550 GMAC/s on one core, 6.0 to 6.7 times the
  previous kernel and 3.9 to 6.1 times ggml's Q8_0 and F16 mul_mat. At a
  clock of about 5 GHz (not measured) that is about 110 int8 multiply-adds
  per cycle, close to two 512-bit `vpdpbusd` per cycle. The gain comes from
  keeping one output row per vector lane (no horizontal reduction per group),
  512-bit vectors, 12 independent int32 accumulators, and unpacking the 2-bit
  weights once per 4 activation rows.
- Per utterance on LibriSpeech, packed ternary is now the fastest form: median
  RTF 75.6 against 46.1 for the same model in F16 and 43.1 for Ultra Q8_0,
  1.6 to 1.8 times faster. The three passes of each row are within 5 percent
  of each other, well inside that margin. WER is unchanged at 1.96 percent.
- On the 180 s clip the gain is small: 7.93 s against 8.73 s (F16) and
  9.04 s (Q8_0), 9 to 12 percent less time. The linears are a smaller share of this run,
  which includes model load and attention over a long sequence.
- The avx2 kernel (184 GMAC/s, 59.0 RTF) is also faster than the ggml F16 and
  Q8_0 paths. It accumulates a whole group in int16, which is exact here.
- Moondream reports 113x for its own Photon runtime with these ternary weights
  on 8 Zen 5 cores, against 45x for parakeet.cpp Q8_0. We did not run Photon;
  our 75.6 is not measured under the same conditions and is not a comparison
  with that number.
- The NEON kernel uses the same layout and is bit-identical to the scalar
  reference under `qemu-aarch64`; its speed on real ARM hardware has not been
  measured.
- The v3 F16 row is 3.8 times slower than Ultra and Redux F16 with identical
  shapes on the long clip. That gap was not investigated.

Transcripts on `tests/fixtures/speech.wav` are identical across the scalar,
avx2 and vnni kernels and the dequantized F16 and Q8_0 Redux, and equal the
reference transcript in `AGENTS.md`.

## Tests

```
ctest --test-dir build -R ternary --output-on-failure        # test_ternary (kernels vs scalar), no model needed

PARAKEET_TEST_GGUF_REDUX_KEEP=<redux --ternary keep gguf> \
PARAKEET_TEST_GGUF_REDUX_DEQ=<redux --ternary dequant gguf> \
    ctest --test-dir build -R test_ternary_model --output-on-failure

build/tests/bench_ternary [N K T reps]                        # single thread: each kernel, the quantizer, ggml Q8_0 and F16 mul_mat
```
