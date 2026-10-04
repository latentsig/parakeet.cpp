# Performance summary

This page keeps the headline speed numbers that the README links to. The full
methodology, all ten models, the plots and the raw results are in
[../benchmarks/BENCHMARK.md](../benchmarks/BENCHMARK.md). Read the method before you
quote a number: the CPU runs used 8 threads on a 20-core x86 host, the GPU runs used
one NVIDIA GB10, and each machine was a shared development host, not a quiet lab box.

RTFx is audio seconds divided by processing seconds. Higher is faster. Speedup is our
RTFx divided by the NeMo (PyTorch) RTFx on the same machine, batch size 1.

## CPU against NeMo (LibriSpeech test-clean, 100 utterances, 8 threads)

| dtype | size vs f32 | mean speedup vs NeMo | accuracy |
| ----- | ----------- | -------------------- | -------- |
| f32   | 100%        | 1.40x (range 1.11x to 1.69x over 10 models) | mean agreement WER 0.015%, byte-identical on most models |
| f16   | 57%         | 1.70x | same as f32 |
| q8_0  | 37%         | 1.56x (up to 1.89x) | mean agreement WER 0.16% |
| q4_k  | 26%         | 1.25x | agreement WER 1.08%, small and monotonic accuracy cost |

Source: the "Quantization" and "Headline" tables in BENCHMARK.md. Peak RAM is roughly
2x lower than NeMo at f32 (for example 2582 MB against 5598 MB for `tdt-0.6b-v3`) and
lower still once quantized.

## GPU against NeMo (NVIDIA GB10, f32, LibriSpeech)

Median speedup over the 10 offline and streaming-EOU models: 1.25x. Best case: 4.3x on
`tdt_ctc-110m`. The smallest gains are on the pure-encoder CTC models (about 1.2x),
because ggml's generic CUDA conv and attention kernels still trail NeMo's tuned cuDNN.
The log-mel front end runs on the GPU through a ggml DFT-matmul graph; the CPU path is
unchanged. NeMo's TDT greedy decode is not CUDA-graph accelerated here and ours is a lean
C++ loop, which explains most of the gap on the TDT and hybrid models.

These numbers come from `benchmarks/results_gpu/` (`scripts/plot_gpu.py` plots them);
the median and maximum above were recomputed from those files when this page was written.
NeMo ran in the `nvcr.io/nvidia/nemo` container for these runs.

## Single clip, newer models

`nemotron-3.5-asr-streaming-0.6b` on a 7.43 s clip (Ryzen 9 9950X3D, 8 threads, median of
7 passes): 2.40x NeMo at f32 and 2.52x at q8_0, byte-identical transcripts. On the GB10 GPU:
1.16x at f32 and 1.30x at q8_0. See the Nemotron section of BENCHMARK.md.

## Against whisper.cpp

The comparison plot [../benchmarks/plots/vs_whisper.png](../benchmarks/plots/vs_whisper.png) shows RTFx and accuracy for parakeet.cpp and whisper.cpp on CPU and GPU (the earlier README text said the 110M Parakeet is faster than whisper base.en and far faster than large-v3-turbo; read the plot for the exact values). The side-by-side clips measure about 12x (GPU) and about 27x (CPU) against whisper.cpp
turbo on one clip with WER 1.6% for both. One clip is a demo, not a benchmark. See
[../benchmarks/media/gpu_whisper_duel.mp4](../benchmarks/media/gpu_whisper_duel.mp4) and
[../benchmarks/media/cpu_duel.mp4](../benchmarks/media/cpu_duel.mp4).

## Apple Metal

On an Apple M4, Metal is about 1.3x to 5.6x faster than CPU, most on the larger models. See
[BENCHMARK.md, Apple Metal](../benchmarks/BENCHMARK.md#apple-metal-m4).

## Batched decode

Up to about 10x to 12x on the GB10 at batch 16 and about 3x to 5x on CPU. See
[batching.md](batching.md).
