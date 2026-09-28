// Sound stream C-API on real audio: ced.cpp's public-domain demo clips
// (rooster, thunder, guitar) back to back, 6 s each. Needs
// PARAKEET_TEST_CED_GGUF (any CED GGUF). Exit 77 when unset.
#include "parakeet_capi.h"
#include "audio_io.hpp"
#include "ced_tagger.hpp"
#include "sound_stream.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Load a WAV as 16 kHz mono (pk::load_audio_16k_mono downmixes and resamples).
static bool load16k(const std::string& path, std::vector<float>& x) {
    pk::Audio a;
    if (!pk::load_audio_16k_mono(path, a)) return false;
    x = std::move(a.samples);
    return true;
}

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL: " __VA_ARGS__); std::fprintf(stderr, "\n"); ++fails; } } while (0)

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_CED_GGUF");
    if (!gguf) { std::fprintf(stderr, "SKIP: PARAKEET_TEST_CED_GGUF unset\n"); return 77; }
    parakeet_ctx* tag = parakeet_capi_load(gguf);
    if (!tag) { std::fprintf(stderr, "FAIL: load\n"); return 1; }

    // Build the clip.
    const char* names[] = {"rooster", "thunder", "guitar"};
    std::vector<float> pcm;
    for (const char* n : names) {
        std::vector<float> x;
        const std::string p = std::string(PK_CED_SOURCE_DIR) + "/benchmarks/demo/clips/" + n + ".wav";
        CHECK(load16k(p, x), "read %s", p.c_str());
        pcm.insert(pcm.end(), x.begin(), x.end());
    }

    // opts_validation
    parakeet_sound_opts o;
    parakeet_capi_sound_opts_default(&o);
    CHECK(o.size == (int)sizeof(o) && o.hop_sec == 1.0f && o.window_sec == 3.0f, "defaults");
    parakeet_sound_opts bad = o;
    bad.hop_sec = 5.0f;
    CHECK(parakeet_capi_sound_stream_begin(tag, &bad) == nullptr, "bad opts accepted");
    CHECK(std::strstr(parakeet_capi_last_error(tag), "hop_sec") != nullptr, "bad opts message: %s",
          parakeet_capi_last_error(tag));

    // Stream it in 0.25 s pieces.
    parakeet_sound_stream* s = parakeet_capi_sound_stream_begin(tag, nullptr);
    CHECK(s != nullptr, "begin: %s", parakeet_capi_last_error(tag));
    std::vector<parakeet_sound_segment> segs;
    std::string scores;
    bool checked_active = false;
    for (size_t i = 0; i < pcm.size(); i += 4000) {
        const int n = (int)std::min<size_t>(4000, pcm.size() - i);
        parakeet_sound_segment* out = nullptr;
        int nout = 0;
        CHECK(parakeet_capi_sound_stream_feed(s, pcm.data() + i, n, i + 4000 >= pcm.size(), &out, &nout) == 0,
              "feed: %s", parakeet_capi_last_error(tag));
        for (int k = 0; k < nout; ++k) segs.push_back(out[k]);
        parakeet_capi_free_sound_segments(out);
        char* j = parakeet_capi_sound_stream_drain_scores_json(s);
        CHECK(j != nullptr, "drain");
        if (j) { scores += j; parakeet_capi_free_string(j); }

        // Around 3.5 s in (inside the rooster slot), check the still-open
        // segments: whatever comes back must be internally consistent, even
        // when nothing happens to be open at this exact instant.
        if (!checked_active && i + n >= 3.5 * 16000) {
            checked_active = true;
            parakeet_sound_segment* act = nullptr;
            int nact = 0;
            CHECK(parakeet_capi_sound_stream_active(s, &act, &nact) == 0,
                  "active: %s", parakeet_capi_last_error(tag));
            const double t = (i + n) / 16000.0;
            for (int k = 0; k < nact; ++k) {
                CHECK(act[k].start <= act[k].end, "active start <= end");
                CHECK(std::fabs(act[k].end - t) < 1e-3, "active end == stream time (%.3f vs %.3f)",
                      act[k].end, t);
            }
            parakeet_capi_free_sound_segments(act);
        }
    }
    for (const auto& g : segs)
        std::fprintf(stderr, "  %-24s %6.2f %6.2f %.3f\n", g.label, g.start, g.end, g.peak);

    // Each clip's main label appears, inside its own 6 s slot (one hop slack).
    auto found = [&](const char* label, float lo, float hi) {
        for (const auto& g : segs)
            if (std::strcmp(g.label, label) == 0 && g.start >= lo - 1.0f && g.start < hi) return true;
        return false;
    };
    CHECK(found("Chicken, rooster", 0.0f, 6.0f), "rooster in [0, 6)");
    CHECK(found("Thunder", 6.0f, 12.0f) || found("Thunderstorm", 6.0f, 12.0f), "thunder in [6, 12)");
    CHECK(found("Guitar", 12.0f, 18.0f) || found("Acoustic guitar", 12.0f, 18.0f), "guitar in [12, 18)");
    CHECK(scores.find("\"tags\":[{\"index\":") != std::string::npos, "score json shape");

    // Window scores are exactly the tagger's numbers for the same samples.
    {
        auto t = pk::CedTagger::load(gguf);
        std::vector<float> direct;
        CHECK(t && t->scorer()(pcm.data(), 16000, direct), "direct score");
        pk::SoundOpts so; so.window_sec = 1.0f; so.hop_sec = 1.0f; so.top_k = 1;
        pk::SoundStream ss(t->scorer(), t->n_classes(), so);
        ss.feed(pcm.data(), 16000, false);
        auto w = ss.drain_windows();
        const float best = *std::max_element(direct.begin(), direct.end());
        CHECK(w.size() == 1 && w[0].top[0].second == best, "window score != direct score");
    }
    // The first window of a 1 s / 1 s stream spans [0, 1).
    {
        parakeet_sound_opts one = o;
        one.window_sec = 1.0f; one.hop_sec = 1.0f; one.top_k = 1;
        parakeet_sound_stream* s1 = parakeet_capi_sound_stream_begin(tag, &one);
        parakeet_sound_segment* out = nullptr; int nout = 0;
        parakeet_capi_sound_stream_feed(s1, pcm.data(), 16000, 0, &out, &nout);
        parakeet_capi_free_sound_segments(out);
        char* j = parakeet_capi_sound_stream_drain_scores_json(s1);
        std::fprintf(stderr, "first window: %s\n", j ? j : "(null)");
        CHECK(j && std::strstr(j, "\"start\":0.000,\"end\":1.000") != nullptr, "first window span");
        parakeet_capi_free_string(j);
        parakeet_capi_sound_stream_free(s1);
    }

    // Finished stream refuses more audio.
    parakeet_sound_segment* out = nullptr; int nout = 0;
    CHECK(parakeet_capi_sound_stream_feed(s, pcm.data(), 10, 0, &out, &nout) != 0, "feed after is_last");
    parakeet_capi_sound_stream_free(s);

    // wrong_ctx_kind: an ASR context is rejected with a clear message.
    if (const char* asr_gguf = std::getenv("PARAKEET_TEST_GGUF")) {
        parakeet_ctx* asr = parakeet_capi_load(asr_gguf);
        CHECK(parakeet_capi_sound_stream_begin(asr, nullptr) == nullptr, "ASR ctx accepted as tagger");
        CHECK(std::strstr(parakeet_capi_last_error(asr), "ASR model") != nullptr, "message: %s",
              parakeet_capi_last_error(asr));
        CHECK(parakeet_capi_num_classes(asr) == -1, "num_classes on ASR ctx");
        parakeet_capi_free(asr);
    }
    CHECK(parakeet_capi_num_classes(tag) == 527, "num_classes");
    parakeet_capi_free(tag);
    if (fails) return 1;
    std::fprintf(stderr, "PASS\n");
    return 0;
}
