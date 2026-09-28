# Sound-event detection

parakeet.cpp can tag everyday sounds (dog bark, glass breaking, applause,
alarms, music, speech, and the rest of the 527-class AudioSet ontology) using
[CED](https://github.com/RicherMans/CED) (Consistent Ensemble Distillation),
run through [ced.cpp](https://github.com/mudler/ced.cpp), the same LocalAI-team
ggml port used standalone. This is separate from ASR: a CED GGUF loads into
its own context (a "tagger"), and `pk::CedTagger` / the sound-stream C-API are
the only code in this repository that talks to it.

CED GGUFs (tiny, mini, small, base; f16 and q8_0, base also f32) are published
in one collection at
[huggingface.co/mudler/ced-gguf](https://huggingface.co/mudler/ced-gguf).
`ced-tiny` at q8_0 is about 6 MB; `ced-base` at q8_0 is about 88 MB.

## Loading a tagger

`parakeet_capi_load` detects the GGUF's architecture and returns a context
that holds either an ASR model, a diarization model, or (for a CED GGUF) a
tagger. Passing a tagger context to the ASR or diarization entry points fails
with a clear "context holds a CED sound model" error, and vice versa.

```c
parakeet_ctx* tagger = parakeet_capi_load("ced-tiny-q8_0.gguf");
```

On the C++ side this is `pk::CedTagger::load(path)`, which exposes
`n_classes()`, `label(index)`, and a `scorer()` (a `pk::SoundScorer`: scores
one PCM window and fills one probability per class, multi-label, not a
softmax).

### Device

ced.cpp picks its own device independently of the rest of parakeet.cpp: on a
GPU build it picks the first GPU it finds and falls back to CPU, and
`CED_DEVICE` overrides that choice the same way `PARAKEET_DEVICE` does for
Parakeet (`CED_DEVICE=cpu` forces CPU; a device name such as `CUDA0`,
`Vulkan0`, or `MTL0` selects that device). `CED_DEVICE` and `PARAKEET_DEVICE`
are read separately, so a tagger and an ASR/diarization context in the same
process can land on different devices.

## Windowing and timing rules

`pk::SoundStream` (`src/sound_stream.hpp`) runs a tagger over live PCM as a
sliding window: every `hop_sec` seconds, the most recent `window_sec` seconds
are scored, and a class's score is compared against two thresholds to decide
whether an event is open:

- A class **opens** when a scored window's score for it is `>= on_threshold`.
- An open class **closes** when a scored window's score drops `< off_threshold`.
- A closed segment shorter than `min_duration_sec` is dropped.

Because a score only tells you "this class was present somewhere in the last
`window_sec`," a segment's start and end are only known to **one hop's**
resolution: the code places the open boundary at the start of the newest hop
inside the triggering window (the latest boundary consistent with the score),
and the close boundary at the end of the oldest hop inside the triggering
window (the earliest boundary consistent with the score). So segment
boundaries can be off by up to one `hop_sec` from the true event edge, never
more.

At end of stream (`is_last`), a final tail window is scored if at least one
CED patch's worth of audio (0.16 s) remains unscored, and every class still
open is closed at the current stream time.

## Defaults

```cpp
struct SoundOpts {
    float window_sec = 3.0f;
    float hop_sec = 1.0f;
    float on_threshold = 0.4f;
    float off_threshold = 0.3f;
    float min_duration_sec = 0.3f;
    int   top_k = 5;
};
```

CED was trained and evaluated on clips up to about 10 s. A live stream cannot
wait 10 s to say what it is hearing, so the question is how much accuracy a
short window gives up. `examples/cli/sound_window_eval.cpp` (built as
`sound-window-eval`) measures this directly: for each WAV, it takes the
whole-clip (up to 10 s) top-1 class as the reference, then slides windows of
1, 2, 3, and 5 s (hop = window / 2) over the same clip and reports the share
of windows whose top-1 matches the reference, and the share whose top-5
contains it.

```
sound-window-eval <ced.gguf> <list.txt>   # one WAV path per line
```

### Measurement

Data: all 2000 clips of [ESC-50](https://github.com/karoldvl/ESC-50) (5 s,
44.1 kHz, resampled to 16 kHz mono) plus the three
`third_party/ced.cpp/benchmarks/demo/clips` demo clips (rooster, thunder,
guitar; 6 s), for `ced-tiny-q8_0`; ESC-50 folds 1 and 2 (800 clips) plus the
same three demo clips for `ced-base-q8_0`. ESC-50 clips are 5 s, so a 5 s
window is the reference clip itself; that row is trivially 100% and is listed
only for shape, not as a real data point.

`ced-tiny-q8_0`, 2003 clips:

| window | windows | top-1 = clip top-1 | clip top-1 in top-5 |
|---|--:|--:|--:|
| 1 s | 18033 | 33.1% | 55.0% |
| 2 s | 8015 | 57.8% | 80.4% |
| 3 s | 4009 | 74.1% | 91.5% |
| 5 s | 2003 | 100.0% | 100.0% |

`ced-base-q8_0`, 803 clips:

| window | windows | top-1 = clip top-1 | clip top-1 in top-5 |
|---|--:|--:|--:|
| 1 s | 7233 | 39.2% | 62.3% |
| 2 s | 3215 | 60.7% | 83.5% |
| 3 s | 1609 | 75.1% | 93.0% |
| 5 s | 803 | 100.0% | 100.0% |

### Decision

The rule was: keep window 3 s / hop 1 s unless ced-base's 3 s windows agree
with the clip top-1 on fewer than 70% of windows, in which case switch to 5 s
/ hop 1 s. ced-base's 3 s row is 75.1%, above the 70% line, so the defaults
stay at window 3 s / hop 1 s. A 1 s window is noticeably worse (39.2% on
ced-base) and a 2 s window is a real step down too (60.7%); 3 s is the
shortest window that stays reasonably close to what the whole clip would say,
which is the point of picking a default for a live stream instead of always
waiting for the whole clip.

## Sound-stream C-API

`include/parakeet_capi.h`, additive since ABI v8:

```c
typedef struct {
    int   size;                    // sizeof(parakeet_sound_opts)
    float window_sec, hop_sec;
    float on_threshold, off_threshold, min_duration_sec;
    int   top_k;
} parakeet_sound_opts;
void parakeet_capi_sound_opts_default(parakeet_sound_opts* o);

typedef struct {
    int         class_index;
    const char* label;             // borrowed from the tagger ctx
    float       start, end, peak;  // seconds from stream start
} parakeet_sound_segment;

typedef struct parakeet_sound_stream parakeet_sound_stream;

parakeet_sound_stream* parakeet_capi_sound_stream_begin(parakeet_ctx* tagger,
                                                        const parakeet_sound_opts* o);
int  parakeet_capi_sound_stream_feed(parakeet_sound_stream* s, const float* pcm, int n,
                                     int is_last, parakeet_sound_segment** out, int* n_out);
int  parakeet_capi_sound_stream_active(parakeet_sound_stream* s,
                                       parakeet_sound_segment** out, int* n_out);
char* parakeet_capi_sound_stream_drain_scores_json(parakeet_sound_stream* s);
void  parakeet_capi_free_sound_segments(parakeet_sound_segment* segs);
void  parakeet_capi_sound_stream_free(parakeet_sound_stream* s);

int         parakeet_capi_num_classes(const parakeet_ctx* ctx);
const char* parakeet_capi_class_label(const parakeet_ctx* ctx, int index);
```

`parakeet_capi_sound_stream_begin` with `o = NULL` uses the defaults above.
`_feed` accepts any chunk size and returns the segments that closed during
that call (free with `parakeet_capi_free_sound_segments`); pass `is_last = 1`
on the final chunk to flush and close everything still open. `_active` reads
the segments still open right now, with `end` set to the current stream time.
`_drain_scores_json` returns the raw per-window top-k scores scored since the
previous drain, as
`[{"start":..,"end":..,"tags":[{"index":..,"label":..,"score":..}]}]`.

### C example

```c
#include "parakeet_capi.h"
#include <stdio.h>

void tag_stream(const char* ced_gguf, const float* pcm, int n_samples) {
    parakeet_ctx* tagger = parakeet_capi_load(ced_gguf);
    if (!tagger) { fprintf(stderr, "load failed\n"); return; }

    parakeet_sound_stream* s = parakeet_capi_sound_stream_begin(tagger, NULL);
    if (!s) { fprintf(stderr, "%s\n", parakeet_capi_last_error(tagger)); parakeet_capi_free(tagger); return; }

    const int chunk = 8000; // 0.5 s at 16 kHz
    for (int i = 0; i < n_samples; i += chunk) {
        const int n = (i + chunk <= n_samples) ? chunk : n_samples - i;
        const int is_last = (i + n >= n_samples);

        parakeet_sound_segment* segs = NULL;
        int n_segs = 0;
        if (parakeet_capi_sound_stream_feed(s, pcm + i, n, is_last, &segs, &n_segs) != 0) {
            fprintf(stderr, "%s\n", parakeet_capi_last_error(tagger));
            break;
        }
        for (int k = 0; k < n_segs; ++k)
            printf("%-24s %6.2f - %6.2f  peak %.2f\n",
                   segs[k].label, segs[k].start, segs[k].end, segs[k].peak);
        parakeet_capi_free_sound_segments(segs);
    }

    parakeet_capi_sound_stream_free(s);
    parakeet_capi_free(tagger);
}
```
