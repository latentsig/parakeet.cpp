// Standalone diarize tool: loads a diarization GGUF and diarizes a WAV.
// Usage: diarize <gguf> <wav> [--stream [model|low|very_low|ultra_low]] [--component NAME]
// <gguf> may be a bundle GGUF: its only "diar" component is used, or the one named by --component.
// Prints {"speakers":N,"segments":[{"speaker","start","end"}, ...]}.
// --stream feeds the audio through the streaming C-API in 100 ms pieces (NeMo
// cache-aware streaming) instead of the whole-file path, optionally in one of
// the model card's latency modes (default: the checkpoint's configuration).
#include "parakeet_capi.h"
#include "audio_io.hpp"
#include "bundle.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int diarize_stream(parakeet_ctx* ctx, const char* wav, int latency) {
    pk::Audio audio;
    if (!pk::load_audio_16k_mono(wav, audio)) {
        std::fprintf(stderr, "cannot read %s\n", wav);
        return 1;
    }
    parakeet_diar_stream* s = parakeet_capi_diarize_stream_begin_latency(ctx, latency);
    if (!s) {
        std::fprintf(stderr, "stream_begin failed: %s\n", parakeet_capi_last_error(ctx));
        return 1;
    }
    std::vector<parakeet_diar_segment> all;
    const int n = (int)audio.samples.size();
    for (int lo = 0; lo < n || lo == 0; lo += 1600) {
        const int len = std::min(1600, n - lo);
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
        std::fprintf(stderr, "usage: %s <gguf> <wav> [--stream [model|low|very_low|ultra_low]]\n", argv[0]);
        return 1;
    }
    std::string component;
    for (int i = 3; i + 1 < argc; ++i)
        if (std::strcmp(argv[i], "--component") == 0) { component = argv[i + 1]; break; }
    const bool stream = argc > 3 && std::strcmp(argv[3], "--stream") == 0;
    int latency = PARAKEET_DIAR_LATENCY_MODEL;
    if (stream && argc > 4 && std::strcmp(argv[4], "--component") != 0) {
        const char* names[] = {"model", "low", "very_low", "ultra_low"};
        latency = -1;
        for (int i = 0; i < 4; ++i)
            if (std::strcmp(argv[4], names[i]) == 0) latency = i;
        if (latency < 0) {
            std::fprintf(stderr, "unknown latency %s (model, low, very_low, ultra_low)\n", argv[4]);
            return 1;
        }
    }
    if (component.empty() && pk::gguf_is_bundle(argv[1])) {
        pk::BundleInfo info;
        std::string err;
        if (!pk::read_bundle_info(argv[1], info, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        int n = 0;
        for (const pk::BundleComponent& c : info.components)
            if (c.kind == pk::kBundleKindDiar) { component = c.name; ++n; }
        if (n != 1) {
            std::fprintf(stderr, "bundle has %s diarization component; name one with --component (components: %s)\n",
                         n == 0 ? "no" : "several", pk::bundle_component_names(info).c_str());
            return 1;
        }
    }
    parakeet_ctx* ctx = component.empty() ? parakeet_capi_load(argv[1])
                                          : parakeet_capi_load_component(argv[1], component.c_str());
    if (!ctx) {
        std::fprintf(stderr, "failed to load %s: %s\n", argv[1], parakeet_capi_load_error());
        return 1;
    }
    if (parakeet_capi_model_kind(ctx) != PARAKEET_MODEL_KIND_DIARIZATION) {
        std::fprintf(stderr, "%s is not a diarization model\n", argv[1]);
        parakeet_capi_free(ctx);
        return 1;
    }
    int rc = 0;
    if (stream) {
        rc = diarize_stream(ctx, argv[2], latency);
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
