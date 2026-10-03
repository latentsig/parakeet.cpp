// VAD-segmented transcription: short audio unchanged, long audio close to the
// single-pass transcript, timestamps monotonic, C-API error path.
// Env: PARAKEET_TEST_GGUF_ULTRA (has a VAD head), PARAKEET_TEST_GGUF (does not; optional).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "audio_io.hpp"
#include "model.hpp"
#include "parakeet_capi.h"

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static std::vector<std::string> words_of(const std::string& s) {
    std::istringstream is(s);
    std::vector<std::string> w;
    for (std::string x; is >> x;) {
        std::string y;
        for (char c : x) if (std::isalnum((unsigned char)c)) y += (char)std::tolower((unsigned char)c);
        if (!y.empty()) w.push_back(y);
    }
    return w;
}
static size_t edit_distance(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    std::vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = j;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= b.size(); ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

int main() {
    const char* ultra = std::getenv("PARAKEET_TEST_GGUF_ULTRA");
    const char* plain = std::getenv("PARAKEET_TEST_GGUF");
    if (!ultra) { std::puts("skip: PARAKEET_TEST_GGUF_ULTRA unset"); return 77; }

    auto m = Model::load(ultra);
    if (!m) { std::fprintf(stderr, "FAIL: load\n"); return 1; }

    // 1. short audio: identical to the plain path
    Audio shortclip;
    CHECK(load_audio_16k_mono("tests/fixtures/speech.wav", shortclip));
    const std::string a = m->transcribe_pcm(shortclip.samples, 16000, Decoder::kTDT);
    const std::string b = m->transcribe_pcm_vad(shortclip.samples, 16000, Decoder::kTDT);
    CHECK(a == b);

    // 2. self-contained long clip (~62 s) from tracked fixtures
    Audio clip;
    {
        const char* parts[] = {"tests/fixtures/two_speakers.wav", "tests/fixtures/speech.wav",
                               "tests/fixtures/two_speakers.wav", "tests/fixtures/speech.wav"};
        for (const char* f : parts) {
            Audio x;
            CHECK(load_audio_16k_mono(f, x));
            clip.samples.insert(clip.samples.end(), x.samples.begin(), x.samples.end());
        }
        clip.sample_rate = 16000;
    }
    const double total = (double)clip.samples.size() / 16000.0;
    {
        SegmenterOpts o;
        o.frame_sec = m->config().vad.frame_sec;
        const auto segs = segment_by_vad(m->vad_probabilities(clip.samples), total, o);
        std::printf("62 s clip (%.2f s): %zu segments:", total, segs.size());
        for (const auto& sg : segs) std::printf(" [%.2f-%.2f]", sg.start, sg.end);
        std::printf("\n");
        CHECK(segs.size() >= 2);
        if (!segs.empty()) {
            CHECK(segs.front().start == 0.0);
            CHECK(std::fabs(segs.back().end - total) < 1e-6);
            for (size_t i = 0; i < segs.size(); ++i) {
                CHECK(segs[i].end - segs[i].start <= o.max_seg_sec + 1e-6);
                if (i + 1 < segs.size()) CHECK(segs[i].end == segs[i + 1].start);
            }
        }
    }
    {
        const std::string full = m->transcribe_pcm(clip.samples, 16000, Decoder::kTDT);
        const std::string seg = m->transcribe_pcm_vad(clip.samples, 16000, Decoder::kTDT);
        const auto wf = words_of(full), ws = words_of(seg);
        const double diff = (double)edit_distance(wf, ws) / (double)std::max<size_t>(1, wf.size());
        std::printf("62 s clip: %zu vs %zu words, word diff %.3f\n", wf.size(), ws.size(), diff);
        CHECK(!ws.empty());
        CHECK(diff < 0.15);
        const Transcription tr = m->transcribe_pcm_vad_with_timestamps(clip.samples, 16000, Decoder::kTDT);
        CHECK(!tr.words.empty());
        float prev = -1.0f;
        for (const Word& w : tr.words) {
            CHECK(w.start >= prev - 1e-3f);
            CHECK(w.end >= w.start);
            prev = w.start;
        }
        if (!tr.words.empty()) CHECK(tr.words.back().end <= (float)total + 0.5f);
        for (size_t i = 1; i < tr.tokens.size(); ++i) CHECK(tr.tokens[i].frame >= tr.tokens[i - 1].frame);
        CHECK(words_of(tr.text) == ws);
    }

    // 2b. The VAD runs on 120 s blocks. A 250 s clip (three blocks) must still
    //     give one probability per 80 ms frame, to within the edge frames.
    {
        Audio big;
        while ((double)big.samples.size() / 16000.0 < 250.0)
            big.samples.insert(big.samples.end(), clip.samples.begin(), clip.samples.end());
        const double dur = (double)big.samples.size() / 16000.0;
        const std::vector<float> p = m->vad_probabilities(big.samples);
        const double fs = m->config().vad.frame_sec;
        std::printf("block grid: %.2f s -> %zu frames (expect about %.1f)\n", dur, p.size(), dur / fs);
        CHECK(std::fabs((double)p.size() - dur / fs) <= 3.0);
        for (float v : p) CHECK(std::isfinite(v) && v >= 0.0f && v <= 1.0f);
    }

    // 3. optional 180 s clip (untracked file)
    Audio longclip;
    if (!load_audio_16k_mono("benchmarks/audio/diverse/i_have_a_dream.wav", longclip)) {
        std::puts("skip long clip: file missing");
    } else {
        const std::string full = m->transcribe_pcm(longclip.samples, 16000, Decoder::kTDT);
        const std::string seg = m->transcribe_pcm_vad(longclip.samples, 16000, Decoder::kTDT);
        const auto wf = words_of(full), ws = words_of(seg);
        const double diff = (double)edit_distance(wf, ws) / (double)std::max<size_t>(1, wf.size());
        std::printf("long clip: %zu vs %zu words, word diff %.3f\n", wf.size(), ws.size(), diff);
        CHECK(!ws.empty());
        CHECK(diff < 0.08);
        const Transcription tr = m->transcribe_pcm_vad_with_timestamps(longclip.samples, 16000, Decoder::kTDT);
        const float dur = (float)longclip.samples.size() / 16000.0f;
        CHECK(!tr.words.empty());
        float prev = -1.0f;
        for (const Word& w : tr.words) {
            CHECK(w.start >= prev - 1e-3f);
            CHECK(w.end >= w.start);
            CHECK(w.end <= dur + 0.5f);
            prev = w.start;
        }
        CHECK(words_of(tr.text) == ws);
    }

    // 4. C-API: JSON works on a VAD model, NULL + message on a model without one
    parakeet_ctx* ctx = parakeet_capi_load(ultra);
    CHECK(ctx != nullptr);
    if (ctx) {
        char* j = parakeet_capi_transcribe_path_json_vad(ctx, "tests/fixtures/speech.wav", 2);
        CHECK(j != nullptr);
        if (j) { CHECK(std::string(j).find("\"words\"") != std::string::npos); parakeet_capi_free_string(j); }
        parakeet_capi_free(ctx);
    }
    if (!plain) std::puts("note: PARAKEET_TEST_GGUF unset, skipping no-VAD-head error check");
    if (plain) {
        auto pm = Model::load(plain);
        CHECK(pm && !pm->config().vad.present);
        parakeet_ctx* c2 = parakeet_capi_load(plain);
        CHECK(c2 != nullptr);
        if (c2) {
            char* j = parakeet_capi_transcribe_path_json_vad(c2, "tests/fixtures/speech.wav", 2);
            CHECK(j == nullptr);
            CHECK(std::string(parakeet_capi_last_error(c2)).find("VAD") != std::string::npos);
            parakeet_capi_free(c2);
        }
    }
    if (failures) return 1;
    std::puts("test_transcribe_vad: OK");
    return 0;
}
