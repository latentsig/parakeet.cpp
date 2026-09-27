// Standalone diarize tool: loads a diarization GGUF and diarizes a WAV.
// Usage: diarize <gguf> <wav> [--stream]
// Prints {"speakers":N,"segments":[{"speaker","start","end"}, ...]}.
// --stream feeds the audio through the streaming C-API in 1 s pieces (NeMo
// cache-aware streaming) instead of the offline path.
#include "parakeet_capi.h"
#include "audio_io.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

static int diarize_stream(parakeet_ctx* ctx, const char* wav) {
    pk::Audio audio;
    if (!pk::load_audio_16k_mono(wav, audio)) {
        std::fprintf(stderr, "cannot read %s\n", wav);
        return 1;
    }
    parakeet_diar_stream* s = parakeet_capi_diarize_stream_begin(ctx);
    if (!s) {
        std::fprintf(stderr, "stream_begin failed: %s\n", parakeet_capi_last_error(ctx));
        return 1;
    }
    std::vector<parakeet_diar_segment> all;
    const int n = (int)audio.samples.size();
    for (int lo = 0; lo < n || lo == 0; lo += 16000) {
        const int len = std::min(16000, n - lo);
        parakeet_diar_segment* segs = nullptr;
        int ns = 0;
        if (parakeet_capi_diarize_stream_feed(s, audio.samples.data() + lo, len,
                                              lo + len >= n, &segs, &ns) != 0) {
            std::fprintf(stderr, "stream_feed failed: %s\n", parakeet_capi_last_error(ctx));
            parakeet_capi_diarize_stream_free(s);
            return 1;
        }
        all.insert(all.end(), segs, segs + ns);
        parakeet_capi_free_diar_segments(segs);
        if (lo + len >= n) break;
    }
    parakeet_capi_diarize_stream_free(s);
    std::sort(all.begin(), all.end(), [](const auto& a, const auto& b) {
        return a.start != b.start ? a.start < b.start : a.speaker < b.speaker;
    });
    std::printf("{\"speakers\":8,\"segments\":[");
    for (size_t i = 0; i < all.size(); ++i)
        std::printf("%s{\"speaker\":%d,\"start\":%.2f,\"end\":%.2f}", i ? "," : "",
                    all[i].speaker, all[i].start, all[i].end);
    std::printf("]}\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <gguf> <wav> [--stream]\n", argv[0]);
        return 1;
    }
    const bool stream = argc > 3 && std::strcmp(argv[3], "--stream") == 0;
    parakeet_ctx* ctx = parakeet_capi_load(argv[1]);
    if (!ctx) {
        std::fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    int rc = 0;
    if (stream) {
        rc = diarize_stream(ctx, argv[2]);
    } else {
        char* json = parakeet_capi_diarize_path(ctx, argv[2]);
        if (!json) {
            std::fprintf(stderr, "diarize failed: %s\n", parakeet_capi_last_error(ctx));
            rc = 1;
        } else {
            std::printf("%s\n", json);
            parakeet_capi_free_string(json);
        }
    }
    parakeet_capi_free(ctx);
    return rc;
}
