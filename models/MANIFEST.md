# parakeet.cpp — Model Publishing Manifest

This file lists the expected set of published GGUF models for parakeet.cpp.
Each row is one source checkpoint × one quantization variant.

The GGUFs themselves are **not committed** (`models/` is git-ignored); only this
manifest is tracked. Run `scripts/publish_hf.py` to produce the GGUFs locally
and (with `--upload`) push them to HuggingFace.

WER (word error rate) is measured against the NeMo reference on
`tests/fixtures/speech.wav` (LibriSpeech `2086-149220-0033`, ~7.4 s, English).
0.0 = byte-for-byte identical transcript. Source: `docs/parity.md` and
`docs/quantization.md`.

---

## Publish command

```bash
# Dry-run (safe — converts locally, prints what would be uploaded, no HF contact)
.venv/bin/python scripts/publish_hf.py --model nvidia/parakeet-tdt_ctc-110m

# Real upload (requires HF token)
.venv/bin/python scripts/publish_hf.py --model nvidia/parakeet-tdt_ctc-110m --upload
```

---

## Expected published set

### `nvidia/parakeet-tdt_ctc-110m` — Hybrid TDT+CTC, 110 M params

| Variant | HF repo | Approx size | WER vs NeMo (TDT) | WER vs NeMo (CTC) | Validated |
|---|---|---:|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-tdt_ctc-110m-f16`  | 255.1 MB | **0.0** | — | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-tdt_ctc-110m-q8_0` | 169.6 MB | **0.0** | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-tdt_ctc-110m-q4_k` | 125.3 MB | **0.0** | — | PASS |

### `nvidia/parakeet-tdt-0.6b-v2` — TDT hybrid, 0.6 B params

| Variant | HF repo | Approx size | WER vs NeMo (TDT) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-tdt-0.6b-v2-f16`  | ~450 MB  | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-tdt-0.6b-v2-q8_0` | 862.0 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-tdt-0.6b-v2-q4_k` | ~350 MB  | not yet measured | — |

### `nvidia/parakeet-tdt-0.6b-v3` — TDT hybrid (multilingual), 0.6 B params

| Variant | HF repo | Approx size | WER vs NeMo (TDT) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-tdt-0.6b-v3-f16`  | ~450 MB | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-tdt-0.6b-v3-q8_0` | ~360 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-tdt-0.6b-v3-q4_k` | ~230 MB | not yet measured | — |

### `nvidia/parakeet-tdt-1.1b` — Pure TDT, 1.1 B params

| Variant | HF repo | Approx size | WER vs NeMo (TDT) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-tdt-1.1b-f16`  | ~750 MB  | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-tdt-1.1b-q8_0` | ~400 MB  | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-tdt-1.1b-q4_k` | ~300 MB  | not yet measured | — |

### `nvidia/parakeet-tdt_ctc-1.1b` — Hybrid TDT+CTC, 1.1 B params

| Variant | HF repo | Approx size | WER vs NeMo | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-tdt_ctc-1.1b-f16`  | ~750 MB | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-tdt_ctc-1.1b-q8_0` | ~400 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-tdt_ctc-1.1b-q4_k` | ~300 MB | not yet measured | — |

### `nvidia/parakeet-ctc-0.6b` — Standalone CTC, 0.6 B params

| Variant | HF repo | Approx size | WER vs NeMo (CTC) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-ctc-0.6b-f16`  | ~450 MB | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-ctc-0.6b-q8_0` | ~240 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-ctc-0.6b-q4_k` | ~160 MB | not yet measured | — |

### `nvidia/parakeet-ctc-1.1b` — Standalone CTC, 1.1 B params

| Variant | HF repo | Approx size | WER vs NeMo (CTC) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-ctc-1.1b-f16`  | ~750 MB | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-ctc-1.1b-q8_0` | ~400 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-ctc-1.1b-q4_k` | ~300 MB | not yet measured | — |

### `nvidia/parakeet-rnnt-0.6b` — Standard RNNT, 0.6 B params

| Variant | HF repo | Approx size | WER vs NeMo (RNNT) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-rnnt-0.6b-f16`  | ~450 MB | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-rnnt-0.6b-q8_0` | ~240 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-rnnt-0.6b-q4_k` | ~160 MB | not yet measured | — |

### `nvidia/parakeet-rnnt-1.1b` — Standard RNNT, 1.1 B params

| Variant | HF repo | Approx size | WER vs NeMo (RNNT) | Validated |
|---|---|---:|---:|---|
| F16  | `mudler/parakeet.cpp-parakeet-rnnt-1.1b-f16`  | ~750 MB | **0.0** | PASS |
| Q8_0 | `mudler/parakeet.cpp-parakeet-rnnt-1.1b-q8_0` | ~400 MB | **0.0** | PASS |
| Q4_K | `mudler/parakeet.cpp-parakeet-rnnt-1.1b-q4_k` | ~300 MB | not yet measured | — |

### `moondream/parakeet-ultra` and `moondream/parakeet-redux` (HF safetensors, v3 shape)

Converted with `scripts/convert_hf_parakeet_to_gguf.py --template <v3 gguf>`.
Published in [mudler/parakeet-cpp-gguf](https://huggingface.co/mudler/parakeet-cpp-gguf) (CC-BY-4.0: credit Moondream and NVIDIA, and note the files are converted copies). Transcript on
`tests/fixtures/speech.wav` matches the reference. WER on LibriSpeech-100 and on synthetic long-form clips (with and without `--vad`) is in `docs/ternary.md`.

| Model | Variant | Converter flags | Size | Notes |
|---|---|---|---:|---|
| parakeet-ultra | F16 | `--dtype f16` | 1441.9 MB | ordinary v3-shaped GGUF |
| parakeet-redux | packed ternary | `--ternary keep` | 213.3 MB | CPU only, no streaming, see `docs/ternary.md` |
| parakeet-redux | dequantized F16 | `--ternary dequant --dtype f16` | 1441.9 MB | runs on any backend |

Published files (sizes in bytes and SHA-256 from the Hugging Face listing):

| File | Source and flags | Size (bytes) | SHA-256 |
|---|---|---:|---|
| [`ultra-f16.gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/ultra-f16.gguf) | ultra, `--dtype f16` | 1,441,900,448 | `5414aea5536178726e8dca20cdfc91cfa3e5443a83a65c4d08acd1c5df65323f` |
| [`ultra-q8_0.gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/ultra-q8_0.gguf) | ultra, `--dtype q8_0` | 941,517,728 | `c2fb452a9df468a141012b01c8c168a25ce93f710897c7de6e353c6cc250986a` |
| [`redux-packed.gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/redux-packed.gguf) | redux, `--ternary keep` (packed) | 213,319,296 | `574614b9a4d9f72ab202877a7ad6a1f4bf819dd1d27b9d42dfe8cd429fcebdd5` |
| [`redux-f16.gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/redux-f16.gguf) | redux, `--ternary dequant --dtype f16` | 1,441,900,448 | `f2e2a9c412191a4ecbeb70f0c9b837749c0147489f7f2d36f3a0b4bc5d41cf79` |
| [`redux-q8_0.gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/redux-q8_0.gguf) | redux, `--ternary dequant --dtype q8_0` | 941,517,728 | `3429a4598946c406dbb4af5bb589d4dd307b069d550d98e24bef6779c4e6988f` |

The packed Redux file is CPU only and offline only. The Redux F16 and Q8_0 files are dequantized
(ordinary weights expanded from the ternary ones). Use a dequantized file for any GPU backend and for CPUs that would run the packed
file on the slow scalar kernel: MSVC builds, Windows on ARM and aarch64 without dotprod. x86-64 with AVX2
or AVX-512 VNNI and aarch64 with dotprod get SIMD kernels for the packed file. The packed file also keeps the
original packed tensors resident next to the repacked planes, so its memory use is more than 213 MB.
Ultra Q8_0 is measured in `docs/ternary.md`.

Both carry the `parakeet.vad.*` KVs and `vad_head.*` tensors unless converted
with `--vad drop`.

---

## Notes

- `parakeet_realtime_eou_120m-v1` (streaming + EOU) is **not** in this manifest — it is
  deferred to Phase 5 (streaming + EOU support).
- `nvidia/parakeet-tdt-0.6b` (v1) is **not published** on HuggingFace (404/401); it was
  superseded by `parakeet-tdt-0.6b-v2`. See `docs/parity.md`.
- Approximate sizes for the 0.6 B / 1.1 B models are estimates based on the 110m
  compression ratios; exact sizes will be populated after conversion.
- Q4_K WER for models other than the 110m anchor has not yet been measured (the 110m
  measured WER 0.0 — see `docs/quantization.md`).

### `snakers4/silero-vad` v6.2.3 (voice activity detection, MIT)

Converted with `scripts/convert_silero_vad_to_gguf.py` from the official
`silero_vad.onnx` (both sample rates in one file). Published in
[mudler/parakeet-cpp-gguf](https://huggingface.co/mudler/parakeet-cpp-gguf).
The model is MIT licensed, Copyright (c) 2020-present Silero Team
(<https://github.com/snakers4/silero-vad>). Max probability difference vs
onnxruntime on the test clip, in `docs/vad.md`.

| Variant | File | Size (bytes) | SHA-256 | Max diff vs onnxruntime | Validated |
|---|---|---:|---|---:|---|
| F32 | [`silero-vad-f32.gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/silero-vad-f32.gguf) | 2,184,480 | `1398e5ce230bd8f20c06f825ed3a41dc4528ef7e6fe8f8ec8a5b7aedefbee943` | 2e-6 | PASS |
| F16 | [`silero-vad-f16.gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/silero-vad-f16.gguf) | 1,264,928 | `8160489282352accc0e95925c2f6bf3d76fb8f7bccdce5c7e46808cb15e8443c` | 3e-3 | PASS |
