# Speaker diarization

parakeet.cpp runs [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization),
a Sortformer model that answers "who spoke when" for up to 8 speakers, and
combines it with any Parakeet ASR model for speaker-attributed transcripts
("who said what").

## Model

- Encoder: FeatureStacking (8 mel frames stacked, 80 ms per step) and a
  31-layer pre-norm Transformer with RoPE attention (d_model 512, 8 heads).
- Head: projection to 192, a subpixel Conv1d that upsamples 8x back to 10 ms
  frames, two linear layers and a sigmoid per speaker.
- Output: per-speaker activity probabilities every 10 ms. Segments come from
  thresholding at 0.5.
- Speakers are numbered in order of first appearance.

## Converting

```
.venv/bin/python scripts/convert_parakeet_to_gguf.py \
    --model nvidia/Nemotron-3-Diarization --dtype q8_0 \
    --output models/nemotron-3-diarization.q8_0.gguf
```

`--model` also takes a local `.nemo`. The converter reads the checkpoint
directly, so it works with a NeMo that cannot instantiate the model (NeMo 3.0
has no RoPE Transformer encoder). Sizes: F32 397 MB, F16 201 MB, Q8_0 109 MB.

## Offline and streaming inference

NeMo's `diarize()` for this checkpoint runs cache-aware streaming inference
(`streaming_mode: true` in the model config), and `parakeet_capi_diarize_*`
does the same: audio is processed in 264-step chunks (21.12 s) with a
speaker cache of 264 steps that keeps speaker identities stable across chunks.
This is also what keeps long recordings correct: the model is trained on
sessions of about 105 s, and attending over a whole long clip at once is
outside that range. On a 12 minute, 3 speaker recording the offline path
(which matches NeMo offline on short clips) agrees with NeMo's `diarize()` on
22% of speech frames and puts almost everything on one speaker; the streaming
path agrees on 100%.

`DiarizationModel::run_offline` / `speaker_probs` still implement the offline
path (NeMo peak-normalizes the waveform there) for short clips and for parity
checks.

## Live streaming and latency modes

The live streaming API (`parakeet_capi_diarize_stream_*`) takes 16 kHz PCM in
pieces of any size and computes the log-mel incrementally (bit-identical to
the whole-clip mel). Segments that continue past a chunk boundary are not
split.

The checkpoint's own configuration processes 21.12 s chunks, too slow for live
labels. The same checkpoint also supports the model card's low-latency
configurations, selected with `parakeet_capi_diarize_stream_begin_latency`:

| Mode | Input latency | Chunk | Look-ahead | FIFO | Cache | Update period |
|---|---|---|---|---|---|---|
| `PARAKEET_DIAR_LATENCY_MODEL` | 21.12 s | 264 | 0 | 0 | 264 | 264 |
| `PARAKEET_DIAR_LATENCY_LOW` | 1.04 s | 9 | 4 | 264 | 264 | 222 |
| `PARAKEET_DIAR_LATENCY_VERY_LOW` | 0.64 s | 6 | 2 | 264 | 264 | 222 |
| `PARAKEET_DIAR_LATENCY_ULTRA_LOW` | 0.32 s | 3 | 1 | 264 | 264 | 222 |

Sizes are 80 ms encoder frames; input latency is (chunk + look-ahead) x 80 ms.
Each chunk is encoded together with its look-ahead, as NeMo's
`streaming_feat_loader` does, and only the chunk itself is output and enters
the FIFO and speaker cache. `parakeet_capi_diarize_stream_active` returns the
segments still open ("who is speaking now") and
`parakeet_capi_diarize_stream_time` how much audio has been diarized.

## Parity with NeMo

Measured against NeMo main (the reference must support the RoPE encoder), with
`scripts/gen_diar_baseline.py`:

| Clip | Speakers | Offline prob max diff | Offline segments | Streaming segments |
|---|---|---|---|---|
| `tests/fixtures/two_speakers.wav`, 23.6 s | 2 | 0.004 | 5 / 5 identical | 5 / 5 identical |
| synthetic two-voice dialogue (VibeVoice sample), 68.5 s | 2 | 0.004 | 26 / 26 identical | 26 / 26 identical |
| synthetic three-voice dialogue (VibeVoice sample), 12.3 min | 3 | | | 100% frame agreement |

Every latency mode matches NeMo in the same mode: on the 23.6 s fixture all
four modes give NeMo's segments exactly (5, 5, 6 and 7 segments), and on the
68.5 s clip the 1.04 s mode, which fills the FIFO and compresses the speaker
cache, gives all 26 segments; probability max diff 0.01 or less throughout.

Segment boundaries match to the 10 ms frame. F16 gives the same results. Q8_0
keeps the same segments on the fixture and the 68.5 s clip; its probabilities
move up to about 0.06 in the checkpoint mode and 0.12 in the low-latency modes
there, where quantization noise changes which frames the cache keeps. Where a
probability sits right at the 0.5 threshold, Q8_0 can also flip a frame or two:
on a 31.5 s two-speaker clip that splits one segment into three short pieces
(99.5% frame agreement). Use F16 when segment-exact output matters.

After the speaker cache compresses, NeMo picks cache frames with
`torch.topk`, whose order for tied scores is arbitrary. parakeet.cpp breaks ties
toward the earlier frame, so streaming probabilities can drift by up to about
0.02 on long clips while the segments stay the same.

## Speaker-attributed ASR

`parakeet_capi_transcribe_and_diarize(_json)` runs an ASR context and a
diarization context on the same PCM and assigns every ASR word to the speaker
whose segments overlap it most. A word that overlaps no segment takes the
nearest segment's speaker if that segment is within 0.5 s, otherwise -1.
Consecutive words from one speaker (gaps up to 0.5 s) form an utterance.

`parakeet_capi_sas_stream_*` does the same live (`_begin_latency` takes a
latency mode): once at least 4 s of diarized audio is uncommitted it is
transcribed, and words that end at least 1 s before the diarized edge are
committed with their speakers. Transcription resumes right after the last
committed word, so no word is cut at the edge and audio the ASR skipped is
heard again with more context. Offline ASR on shorter windows loses words, so
committed text lags by about 3 to 5 s whatever the diarization latency.

## Speed

Streaming modes on the 12.3 min, 3 speaker clip (model load included; agreement
is with NeMo's default `diarize()`, so it shows what the lower latency costs):

| Mode | F16 | x real time | Agreement | Q8_0 | Agreement |
|---|---|---|---|---|---|
| checkpoint (21.12 s) | 5.7 s | 130x | 100% | 6.0 s | 99.9% |
| low (1.04 s) | 107 s | 6.9x | 99.4% | 115 s | 99.4% |
| very low (0.64 s) | 166 s | 4.4x | 99.1% | 172 s | 99.2% |
| ultra low (0.32 s) | 295 s | 2.5x | 98.3% | 332 s | 98.4% |

A low-latency chunk costs about 0.1 s on this CPU, so labels arrive about
0.1 s after the input latency.

Whole-file diarization:

End to end with the `diarize` example (model load included), AMD Ryzen 9
9950X3D, CPU:

| Audio | F32 | Q8_0 |
|---|---|---|
| 23.6 s | 0.43 s | 0.25 s |
| 68.5 s | 0.80 s | 0.60 s |
| 12.3 min | 6.8 s | 6.0 s |

Streaming cost grows linearly with length (each chunk attends over at most
528 steps), so long recordings run at about 110x real time.

## Tests

```
PARAKEET_TEST_DIAR_GGUF=diar.gguf PARAKEET_TEST_BASELINE_DIAR=diar_baseline.gguf \
PARAKEET_TEST_GGUF=asr.gguf ctest --test-dir build -R "diar|sas|combined"
```

- `test_diarization_accuracy`: offline probabilities and segments, and the
  default `diarize_pcm` segments, against NeMo.
- `test_streaming_diarization`: streaming probabilities and segments against
  NeMo streaming, from a whole-clip mel and from live 100 ms PCM pieces.
- `test_combined_offline`: speaker-attributed ASR and the streaming C-API on
  the two-speaker fixture.
- `test_sas_merge`: word to speaker assignment (no model needed).

Set `PARAKEET_TEST_DIAR_PROB_TOL=0.15` for Q8_0 (the segment checks stay
exact). The baseline includes every latency mode by default; NeMo runs the
low-latency modes slowly on CPU, so `--modes low` limits a long clip to one.
