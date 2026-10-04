# VAD benchmark scripts

Scripts behind [docs/vad-benchmarks.md](../../docs/vad-benchmarks.md). The page says what
each number means and which caveats apply. No audio and no model files are committed;
`results/` holds the small result files of the runs on that page.

All paths are arguments. Run from the repository root. `PARAKEET_LIB` is the path of
`libparakeet.so` (build with `-DPARAKEET_SHARED=ON`; the default is `build/libparakeet.so`).

Python packages: `numpy soundfile librosa datasets torch onnxruntime silero-vad`
(the Silero reference is `silero-vad` 6.2.3). Models: the ASR GGUFs of
`moondream/parakeet-ultra` (Q8_0 and F16), `moondream/parakeet-redux` (packed `keep` and
dequantized F16), a TDT 0.6B v3 F16 GGUF for the reference of the TED talks, and the
Silero GGUFs made with `scripts/convert_silero_vad_to_gguf.py`.

## Data

| Script | What |
| --- | --- |
| `fetch_data.py OUT [--libri-count N] [--ted-seconds S]` | LibriSpeech test-clean (every 13th utterance) as `libri.npy`, and the first 600 s of one TED-LIUM talk |
| `fetch_ted_talks.py OUT [N]` | the first N TED-LIUM long-form talks under 1500 s as `ted_<name>.wav` |
| `make_clips.py DATA` | the 120 synthetic clips (clean, white 10 dB, pink 10 dB) of the Silero parity run |
| `vad_synth.py` | clip builder, noise, and metrics (frame F1, kappa, boundary error) |

## Parakeet heads against Silero

```
scripts/vad_bench/compare_synthetic.py DATA GGUF_DIR out/synthetic_raw.json 24
scripts/vad_bench/analyze_synthetic.py out/synthetic_raw.json
scripts/vad_bench/ted_reference.py DATA tdt-0.6b-v3-f16.gguf      # words of each talk
scripts/vad_bench/compare_ted.py DATA GGUF_DIR out/ted_metrics.json
taskset -c 0-7 scripts/vad_bench/speed_heads.py talk.wav GGUF_DIR   # timing, quiet machine only
```

`GGUF_DIR` holds `ultra-q8_0.gguf`, `ultra-f16.gguf`, `redux-keep.gguf` and
`redux-deq-f16.gguf`. `compare_ted.py` prints speeds; ignore them, use `speed_heads.py`.

## Silero in whisper.cpp and in parakeet.cpp

Build whisper.cpp (CPU), then the test program, which uses its public `whisper_vad_*` API:

```
g++ -O2 -std=c++17 scripts/vad_bench/wvad.cpp -I whisper.cpp/include -I whisper.cpp/ggml/include \
    -L whisper.cpp/build/bin -lwhisper -Wl,-rpath,$PWD/whisper.cpp/build/bin -o wvad
python3 scripts/vad_bench/silero_collect.py CORE DATA MODEL_DIR build/examples/cli/parakeet-cli ./wvad
python3 scripts/vad_bench/silero_analyze.py DATA      # run where probs.npz was written
python3 scripts/vad_bench/speed_silero.py CORE ROUNDS DATA/ted300.wav MODEL_DIR \
    build/examples/cli/parakeet-cli ./wvad whisper.cpp/build/bin out/speed.json
```

`ted300.wav` is the first 300 s of `ted_talk.wav`. `MODEL_DIR` holds `silero_vad.onnx`,
`silero-vad-f32.gguf`, `silero-vad-f16.gguf`, `ggml-silero-v6.2.3-ggml.bin`,
`ggml-silero-v6.2.0.bin` and, for the end to end job, `ggml-tiny.en.bin`.
`silero_analyze.py` expects `probs.npz` and `segs_own_default.json` in the current directory.

## Long talks, batched decode

```
scripts/vad_bench/longform_b1.sh TALKS_DIR GGUF_DIR BUILDS_DIR out.tsv loadlog.txt
```

`BUILDS_DIR` has `build-base`, `build-b0`, `build-b1`: builds of the three commits to compare.

## Timing rules

Timing scripts serialize on a lock file (`/tmp/pk-bench.lock`, `flock`), pin cores with
`taskset`, and record the load average at the start of every run. Run them only on a
quiet machine, and keep the load log with the result. Do not quote a timing from a
loaded machine as a benchmark.

## Slice only head (PR 87)

`vad_slice_bench.cpp` is built inside the PR 87 tree (it uses internal headers) and driven
by `slice_bench.py speed|load|rss`; see its docstring for the environment variables.
Results are in `results/slice_*`.
