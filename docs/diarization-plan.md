# Plan: Nemotron-3-Diarization support in parakeet.cpp

## What the model is

Nemotron-3-Diarization is a Sortformer speaker diarization model
(`SortformerEncLabelModel` in NeMo). It determines "who spoke when" in
audio with up to 8 speakers. Two stages:

1. **NEST FastConformer encoder** — 16kHz audio → log-mel → `[d_model, T]`.
   Architecturally the same FastConformer parakeet.cpp already runs.
2. **Transformer encoder sorting head** — takes encoder output → per-frame,
   per-speaker sigmoid logits. No text, no tokenizer, no CTC/RNNT/TDT decoder.

Streaming uses **AOSC (Arrival-Order Speaker Cache) + FIFO queue** — a
different streaming mechanism from NeMo's cache-aware streaming that
parakeet.cpp currently implements. Input buffer latency from 80ms
(ultra-low-latency) to 30.4s (offline-style); output frame resolution
configurable in multiples of 10ms.

## What's reusable as-is (~60-70% of the engine)

The FastConformer encoder, mel frontend, subsampling, positional encoding,
relpos attention, conformer layers, ggml graph infrastructure, audio I/O,
and FFT are all cleanly separable from the ASR-specific heads. The encoder
emits a `[d_model, Tout]` channels-first tensor — a diarization head plugs
in at the same boundary as `CTCDecoder` (`src/ctc_decoder.cpp`).

| Component | File(s) | Reuse |
|---|---|---|
| Mel frontend (offline) | `src/mel.cpp` (`MelFrontend`, `MelKernel`) | As-is |
| Mel frontend (GPU) | `src/mel_gpu.cpp` (`GpuMel`) | As-is |
| Mel frontend (streaming) | `src/mel.cpp` (`StreamingMel`) | As-is (if `normalize=NA`) |
| Subsampling | `src/subsampling.cpp` | As-is |
| Positional encoding | `src/pos_enc.cpp` | As-is |
| RelPos attention | `src/relpos_attention.cpp` | As-is |
| Conformer layer | `src/conformer.cpp` | As-is |
| FastConformer encoder | `src/encoder.cpp` | As-is |
| GGML graph infra | `src/ggml_graph.cpp`, `src/graph_builder.hpp` | As-is |
| Audio I/O | `src/audio_io.cpp` | As-is |
| FFT | `src/fft.cpp` | As-is |

## Where ASR-specific assumptions are baked in

| Location | Assumption | Fix |
|---|---|---|
| `src/model_loader.cpp:203` | `return cfg_.d_model>0 && cfg_.vocab_size>0` | Relax to `cfg_.d_model>0` or branch on arch |
| `scripts/convert_parakeet_to_gguf.py:38` | `from nemo.collections.asr.models import ASRModel` | Also accept `SortformerEncLabelModel` |
| `scripts/convert_parakeet_to_gguf.py:297-302` | Vocab/tokenizer emitted unconditionally | Make conditional on arch |
| `src/model.cpp` (entire `Model` class) | All methods return text | New `DiarizationModel` class |
| `include/parakeet_capi.h` | All entry points return transcripts | New `diarize_*` surface |
| `src/streaming_encoder.cpp:40,56-58` | Hard-asserts NeMo cache-aware streaming | New AOSC+FIFO streaming path |
| `tests/test_model_loader.cpp:21-22` | Asserts `vocab_size > 0` | Conditional on arch |

## Architectural decisions (decide upfront, before Phase 1)

### Decision 1: Separate `DiarizationModel` class, not bolt-on to `Model`

The existing `Model` class (`src/model.hpp:22-119`) is entirely ASR-shaped —
every public method returns `std::string` or `Transcription`. Bolting
diarization onto it would pollute the class. A separate `DiarizationModel`
class composes the same reusable pieces (`MelFrontend`, `Encoder`, new
`DiarizationHead`) — mirroring how `StreamingSession` is already a separate
class from `Model`.

### Decision 2: Shared segment/word timestamp types

Both ASR and diarization produce timestamped outputs. Design the types so
they compose:

```cpp
// Existing (src/transcription.hpp):
struct Word { std::string text; float start; float end; float conf; };

// New (src/diarization.hpp):
struct SpeakerSegment {
    int   speaker;    // 0-indexed speaker label
    float start;      // seconds
    float end;        // seconds
    float conf;       // aggregate per-frame confidence
};
```

Both use `float start/end` in seconds. The merge for combined ASR+diarization
is then: for each word's `[start, end]`, find the dominant speaker in the
overlapping segments. This is a timestamp intersection, not a deep
architecture coupling.

### Decision 3: Mel computation stays composable

Don't bake mel computation into `DiarizationModel`. Keep `MelFrontend` as a
standalone composable step (as it already is) so both models can share it
when running combined ASR+diarization on the same audio.

### Decision 4: C-API designed for composition from the start

The diarization C-API uses a separate `parakeet_diar_ctx` opaque type, not
overloading `parakeet_ctx`. This lets a caller hold both contexts and call
both APIs, and later call a combined `parakeet_capi_transcribe_and_diarize`
that takes both.

### Decision 5: New arch string `"diarization"`

Add `arch = "diarization"` to the arch vocabulary. The model loader dispatches
on this to know it's not an ASR model (skip vocab/decoder/joint loading, load
diarization head config instead).

---

## Phase 1: Offline diarization (standalone)

**Goal**: Load Nemotron-3-Diarization GGUF, run offline diarization on a WAV
file, produce speaker segments. No streaming.

### 1.1 Converter changes (`scripts/convert_parakeet_to_gguf.py`)

- Import `SortformerEncLabelModel` alongside `ASRModel`.
- Add `"diarization"` branch to `detect_arch()`.
- Make vocab/tokenizer emission conditional — skip for diarization (no
  tokenizer in a diarization checkpoint; `m.tokenizer` would `AttributeError`).
- Emit new GGUF KV:
  - `parakeet.diarization.num_speakers` (max 8)
  - `parakeet.diarization.threshold` (sigmoid threshold, default 0.5)
  - `parakeet.diarization.head_layers` (transformer encoder layer count)
  - `parakeet.diarization.head_d_model`
  - `parakeet.diarization.head_n_heads`
  - `parakeet.diarization.head_ff_dim`
- The generic tensor loop (line 329-357) writes NeMo state_dict keys verbatim
  — encoder tensors (`encoder.layers.N.*`, `encoder.pre_encode.*`) convert
  with zero changes. Add Sortformer-head linear patterns to the quantization
  allowlist (`_QUANTIZABLE_PATTERNS`).
- Featurizer buffer lift (`preprocessor.featurizer.fb`, `.window`) is already
  generic — works as-is.

### 1.2 Model loader changes

- `src/model_loader.cpp:203`: relax `vocab_size>0` check to
  `cfg_.d_model>0` (or branch: `arch == "diarization"` → skip vocab check).
- Add `DiarizationCfg` sub-struct to `ParakeetConfig` in
  `src/model_loader.hpp`:
  ```cpp
  struct DiarizationCfg {
      uint32_t num_speakers = 0;
      float threshold = 0.5f;
      uint32_t head_layers = 0;
      uint32_t head_d_model = 0;
      uint32_t head_n_heads = 0;
      uint32_t head_ff_dim = 0;
      bool present = false;
  };
  ```
- Read `parakeet.diarization.*` KV in `ModelLoader::load`.
- The `parakeet.decoder.*` / `parakeet.joint.*` / `parakeet.tdt.*` fields
  already default to 0 when absent — safe for diarization GGUFs that omit
  them.

### 1.3 Diarization head (`src/diarization_head.hpp` / `.cpp`)

New file, mirrors `src/ctc_decoder.hpp`/`.cpp` as a template:

- `class DiarizationHead`:
  - `DiarizationHead(const ModelLoader& ml)` — reads transformer encoder
    config + weights.
  - `void forward(const std::vector<float>& enc, int d_model, int T,
    std::vector<float>& probs, int& num_speakers)` — takes `[d_model, T]`,
    runs transformer encoder layers (standard MHSA, not relpos) + sigmoid
    output layer, returns `[T, num_speakers]` per-frame speaker probabilities.
- The transformer encoder uses standard multi-head self-attention (not
  relpos). parakeet.cpp currently only has `RelPosAttention` — need a plain
  `MultiHeadAttention` or verify if the Sortformer head uses a different
  attention variant. Check NeMo source for the exact attention type.
- Weight names: NeMo keys like
  `sortformer_modules.transformer_encoder.layers.N.*`,
  `sortformer_modules.encoder2unfold.*`,
  `sortformer_modules.linear_layer.*`.

### 1.4 DiarizationModel class (`src/diarization.hpp` / `.cpp`)

New class, composes `MelFrontend` + `Encoder` + `DiarizationHead`:

```cpp
class DiarizationModel {
public:
    static std::unique_ptr<DiarizationModel> load(const std::string& gguf_path);
    std::vector<SpeakerSegment> diarize_pcm(
        const std::vector<float>& pcm, int sample_rate) const;
    std::vector<SpeakerSegment> diarize_path(const std::string& wav_path) const;
private:
    ModelLoader loader_;
};
```

Orchestration: `pcm → resample to 16k → MelFrontend → Encoder → DiarizationHead
→ threshold per-frame sigmoid → merge consecutive frames with same active
speaker → segments`.

The merge logic: threshold the per-frame per-speaker probabilities at
`cfg.diarization.threshold`, group consecutive frames where the same speaker
is active into segments, convert frame indices to seconds using `frame_sec`
(`hop_length * subsampling_factor / sample_rate`).

### 1.5 C-API surface (`include/parakeet_capi.h`, `src/parakeet_capi.cpp`)

New entry points (bump ABI version v5 → v6):

```c
typedef struct parakeet_diar_ctx parakeet_diar_ctx;

typedef struct parakeet_segment {
    int   speaker;
    float start;
    float end;
    float conf;
} parakeet_segment;

parakeet_diar_ctx* parakeet_capi_diar_load(const char* gguf_path);
void parakeet_capi_diar_free(parakeet_diar_ctx* ctx);

parakeet_segment* parakeet_capi_diarize_path(
    parakeet_diar_ctx* ctx, const char* wav_path, int* n_segments);
parakeet_segment* parakeet_capi_diarize_pcm(
    parakeet_diar_ctx* ctx, const float* samples, int n_samples,
    int sample_rate, int* n_segments);
void parakeet_capi_free_segments(parakeet_segment* segs);

// JSON variant:
// {"segments":[{"speaker":0,"start":0.48,"end":2.16,"conf":0.91},...],
//  "frame_sec":0.080000}
char* parakeet_capi_diarize_path_json(parakeet_diar_ctx* ctx, const char* wav_path);
char* parakeet_capi_diarize_pcm_json(parakeet_diar_ctx* ctx,
    const float* samples, int n_samples, int sample_rate);
```

### 1.6 Tests

- Relax `tests/test_model_loader.cpp:21-22` to not assert `vocab_size > 0`
  when `arch == "diarization"`.
- `tests/test_diarization_head.cpp` — unit test: encoder output → head →
  per-frame speaker probs, compare vs NeMo baseline `.npz`.
- `tests/test_diarization.cpp` — end-to-end: PCM → segments, compare vs NeMo
  `diar_model.diarize()` baseline.
- Existing encoder/mel/conformer/subsampling tests carry over unchanged
  (they're arch-agnostic, test components in isolation).

### 1.7 Parity validation

- Set up NeMo baseline: load `SortformerEncLabelModel.from_pretrained(...)`,
  run `diar_model.diarize(audio=[...])`, dump segments as `.json` baseline.
- Dump intermediate tensors (mel, encoder_out, head_probs) as `.npz` for
  per-component parity testing.
- Match NeMo's segment merging logic (consecutive frames, same speaker,
  threshold).

---

## Phase 2: Streaming diarization (AOSC + FIFO)

**Goal**: Stream audio in chunks, get incremental speaker segments with low
latency (80ms minimum, 0.32s recommended).

### 2.1 Sortformer streaming encoder (`src/sortformer_streaming.hpp` / `.cpp`)

The existing `StreamingEncoder` (`src/streaming_encoder.cpp`) is NOT reusable
— it hard-asserts NeMo cache-aware streaming at the ctor (lines 40, 56-58):
`c.streaming.present`, `c.causal_downsampling`, `c.conv_causal`,
`att_context_style == "chunked_limited"`. Sortformer uses a fundamentally
different streaming mechanism (AOSC + FIFO).

**Reusable from existing streaming code:**
- Graph-input/cache-capture pattern (`graph_input_tensor`, `capture_graph_output`)
- `run_graph` + `GraphInputPool` machinery
- Subsampling `in_valid_frames` override path

**New (not reusable):**
- AOSC cache: retains speaker summary representations (not raw K/V columns)
- FIFO queue: manages chunk overlap/drop
- Attention cache structure (different from NeMo's conv-left-context + K/V cache)
- The conformer layer's `build_stream_layer` cache threading is the wrong
  mechanism for AOSC

### 2.2 Streaming C-API

```c
typedef struct parakeet_diar_stream parakeet_diar_stream;

parakeet_diar_stream* parakeet_capi_diar_stream_begin(parakeet_diar_ctx* ctx);

// Feed PCM, get newly-finalized segments
parakeet_segment* parakeet_capi_diar_stream_feed(
    parakeet_diar_stream* s, const float* pcm, int n_samples,
    int* n_new_segments);

// Flush remaining audio, get tail segments
parakeet_segment* parakeet_capi_diar_stream_finalize(
    parakeet_diar_stream* s, int* n_tail_segments);

void parakeet_capi_diar_stream_free(parakeet_diar_stream* s);
```

### 2.3 Streaming config in GGUF

New GGUF KV for Sortformer streaming:
- `parakeet.sortformer.chunk_len` (frames, e.g. 340 = 27.2s at 80ms/frame)
- `parakeet.sortformer.chunk_right_context` (frames, e.g. 40)
- `parakeet.sortformer.fifo_len` (frames, e.g. 40)
- `parakeet.sortformer.spkcache_len` (AOSC cache size in frames)
- `parakeet.sortformer.spkcache_update_period` (frames, e.g. 300)

### 2.4 Tests

- `tests/test_sortformer_streaming.cpp` — streaming parity vs NeMo streaming
  config baseline.
- Test chunk boundary correctness (no speaker label jumps at chunk edges).
- Test AOSC persistence (speaker identity maintained across chunks).

---

## Phase 3: Speaker-attributed ASR (combined parakeet + diarization)

**Goal**: Run both ASR and diarization, merge into speaker-attributed
transcription ("who said what").

### 3.1 Merge layer (`src/sas_merge.hpp` / `.cpp`)

Simple timestamp intersection:
```cpp
struct SpeakerWord {
    int         speaker;   // from diarization
    std::string text;      // from ASR
    float       start;     // from ASR word
    float       end;       // from ASR word
    float       conf;      // from ASR word
};

std::vector<SpeakerWord> merge_asr_diarization(
    const std::vector<Word>& words,           // ASR (timestamped)
    const std::vector<SpeakerSegment>& segs,  // diarization (timestamped)
    float frame_sec);
```

For each word's `[start, end]`, find the dominant active speaker in the
overlapping diarization segments. This is a linear scan, not a deep
architecture coupling.

### 3.2 Combined C-API

```c
// Load both models, get speaker-attributed transcription
typedef struct parakeet_sas_result {
    int   speaker;
    char* text;
    float start;
    float end;
    float conf;
} parakeet_sas_result;

parakeet_sas_result* parakeet_capi_transcribe_and_diarize(
    parakeet_ctx* asr_ctx,
    parakeet_diar_ctx* diar_ctx,
    const float* samples, int n_samples, int sample_rate,
    int* n_results);
void parakeet_capi_free_sas_results(parakeet_sas_result* results);

// JSON variant with full per-word + per-segment detail
char* parakeet_capi_transcribe_and_diarize_json(
    parakeet_ctx* asr_ctx,
    parakeet_diar_ctx* diar_ctx,
    const float* samples, int n_samples, int sample_rate);
```

### 3.3 Mel sharing optimization

Both models compute log-mel on the same 16kHz audio. If mel configs match
(`n_mels`, `hop_length`, `n_fft`, `preemph`, `mag_power` all identical), compute
mel once and feed both encoders. If configs differ, compute mel twice (cost is
negligible vs two encoder forward passes).

Check at load time whether the two configs are compatible for mel sharing.

### 3.4 Streaming combined ASR + diarization

The two models have different streaming mechanisms and latencies:
- ASR: NeMo cache-aware streaming (chunked-limited attention, conv left-context)
- Diarization: AOSC + FIFO (speaker cache, different chunk structure)

The ASR model might emit a word at time T, but the diarization model's speaker
decision for frame T might not be finalized yet. Need a **merge buffer** that:
1. Holds ASR word hypotheses with timestamps
2. Holds diarization frame labels
3. Emits combined `(speaker, text, start, end)` only when both models have
   committed to a time range

This is a bounded-delay merge problem — doable but requires careful design.

### 3.5 LocalAI integration

Wire the combined ASR+diarization as a LocalAI backend endpoint:
- `/v1/audio/transcriptions` with `diarize=true` → speaker-attributed text
- Streaming variant for real-time use

### 3.6 Tests

- `tests/test_sas_merge.cpp` — unit test the merge logic with known
  word/segment inputs.
- `tests/test_combined_offline.cpp` — end-to-end: audio → both models →
  speaker-attributed transcription, compare vs NeMo SAS baseline.
- `tests/test_combined_streaming.cpp` — streaming combined with merge buffer.

### 3.7 Publish

- Quantize + publish diarization GGUF to HuggingFace.
- Publish combined ASR+diarization documentation.

---

## Dependency graph

```
Phase 1 (offline diarization)
  ├── 1.1 Converter ──────┐
  ├── 1.2 Loader ─────────┤
  ├── 1.3 DiarizationHead ┼── 1.4 DiarizationModel ── 1.5 C-API ── 1.6 Tests ── 1.7 Parity
  └───────────────────────┘

Phase 2 (streaming diarization)
  ├── 2.1 SortformerStreamingEncoder ── 2.2 C-API ── 2.3 GGUF KV ── 2.4 Tests
  └── depends on Phase 1 (head + model + converter)

Phase 3 (combined ASR + diarization)
  ├── 3.1 Merge layer ── 3.2 C-API ── 3.3 Mel sharing ── 3.6 Tests ── 3.7 Publish
  ├── 3.4 Streaming combined (depends on Phase 2)
  ├── 3.5 LocalAI integration
  └── depends on Phase 1 (offline diarization works)
      Phase 2 (for streaming combined)
```

## Risk areas / unknowns

1. **Sortformer transformer head attention type**: Need to verify whether the
   Sortformer sorting head uses standard MHSA, relpos, or a variant. parakeet.cpp
   currently only has `RelPosAttention`. If standard MHSA is needed, it's a new
   ggml graph (not hard, but needs implementing). Check NeMo source for the exact
   attention type.

2. **AOSC cache mechanism**: The AOSC is the novel streaming component. Its exact
   implementation (what is cached, how it's updated, how it interacts with the
   transformer attention) needs to be understood from the NeMo source before
   implementing. This is the highest-risk piece of Phase 2.

3. **Mel config compatibility**: Whether the diarization model's mel config
   matches any existing parakeet model's config (for mel sharing in Phase 3).
   If `normalize="per_feature"`, streaming mel can't be used incrementally —
   `MelFrontend::compute` on the full clip works fine (offline path).

4. **Frame alignment**: ASR and diarization may have different subsampling
   factors, producing different `frame_sec` values. The merge must handle this
   by working in seconds (both produce timestamps in seconds), not frame indices.

5. **Segment merging conventions**: NeMo's segment merging logic (consecutive
   frames, same speaker, threshold, minimum segment duration) needs to be
   matched exactly for parity. Check NeMo `diarize()` output format.
