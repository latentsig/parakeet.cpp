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

The live streaming API (`parakeet_capi_diarize_stream_*`) takes 16 kHz PCM in
pieces of any size, computes the log-mel incrementally (bit-identical to the
whole-clip mel) and returns segments once per 21.12 s chunk. Segments that
continue past a chunk boundary are not split.

## Parity with NeMo

Measured against NeMo main (the reference must support the RoPE encoder), with
`scripts/gen_diar_baseline.py`:

| Clip | Speakers | Offline prob max diff | Offline segments | Streaming segments |
|---|---|---|---|---|
| `tests/fixtures/two_speakers.wav`, 23.6 s | 2 | 0.004 | 5 / 5 identical | 5 / 5 identical |
| synthetic two-voice dialogue (VibeVoice sample), 68.5 s | 2 | 0.004 | 26 / 26 identical | 26 / 26 identical |
| synthetic three-voice dialogue (VibeVoice sample), 12.3 min | 3 | | | 100% frame agreement |

Segment boundaries match to the 10 ms frame. F16 gives the same results; Q8_0
keeps the same segments with a probability max diff of about 0.03.

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

`parakeet_capi_sas_stream_*` does the same live: when a diarization chunk
completes, the audio not yet committed is transcribed, and words that end at
least 1 s before the chunk edge are committed with their speakers. The rest is
transcribed again with the next chunk, so no word is cut at the edge.

## Speed

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

Set `PARAKEET_TEST_DIAR_PROB_TOL=0.05` for Q8_0.
