// The VAD run gate on real detectors. The clip is two_speakers.wav, 1 s of
// digital silence, 12 s of seeded noise, 1 s of silence and speech.wav, so it
// has a noise stretch between two stretches of speech and is longer than the
// 30 s segment cap. Checked per detector (white and pink noise):
//   * "run_gate":0 and no option give the same JSON, byte for byte, in both
//     modes;
//   * in "speech" mode a gate never adds speech;
//   * with the Redux head at 0.92 most of the seconds the head called speech
//     inside the noise stretch are gone, and the speech stretches are unchanged;
//   * with a full ASR head the transcribe path honours the key: no word in the
//     noise stretch, and "run_gate":0 gives the document of no option.
// Env (each optional; skip 77 when none is set):
//   PARAKEET_TEST_VAD_ONLY_REDUX_GGUF  a Redux VAD-only slice (scripts/slice_vad_gguf.py)
//   PARAKEET_TEST_GGUF_REDUX_KEEP / _REDUX_DEQ / _ULTRA  ASR models with a head
//   PARAKEET_TEST_SILERO_GGUF    a Silero VAD model
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "audio_io.hpp"
#include "model.hpp"
#include "parakeet_capi.h"

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

struct Seg { double a, b; };

static std::string take(char* p) {
    if (!p) return std::string();
    std::string s(p);
    parakeet_capi_free_string(p);
    return s;
}

// The "segments" array of a VAD JSON document.
static std::vector<Seg> segs_of(const std::string& j) {
    std::vector<Seg> out;
    size_t i = j.find("\"segments\":[");
    if (i == std::string::npos) return out;
    const size_t end = j.find(']', i);
    for (size_t k = j.find("\"start\":", i); k != std::string::npos && k < end; k = j.find("\"start\":", k + 1)) {
        const double a = std::atof(j.c_str() + k + 8);
        const size_t e = j.find("\"end\":", k);
        out.push_back({a, std::atof(j.c_str() + e + 6)});
    }
    return out;
}

static double overlap(const std::vector<Seg>& s, double lo, double hi) {
    double t = 0.0;
    for (const Seg& g : s) t += std::max(0.0, std::min(g.b, hi) - std::max(g.a, lo));
    return t;
}

static double covered(const std::vector<Seg>& s) { return overlap(s, -1e9, 1e9); }

static bool same(const std::vector<Seg>& a, const std::vector<Seg>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i].a - b[i].a) > 1e-9 || std::fabs(a[i].b - b[i].b) > 1e-9) return false;
    return true;
}

static std::vector<Seg> within(const std::vector<Seg>& s, double lo, double hi) {
    std::vector<Seg> o;
    for (const Seg& g : s) if (g.a >= lo && g.b <= hi) o.push_back(g);
    return o;
}

// White noise, or pink noise from Paul Kellet's filter, scaled to the given RMS.
static std::vector<float> make_noise(bool pink, size_t n, float rms, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(n);
    float b0 = 0, b1 = 0, b2 = 0;
    for (size_t i = 0; i < n; ++i) {
        const float w = nd(rng);
        if (pink) {
            b0 = 0.99765f * b0 + w * 0.0990460f;
            b1 = 0.96300f * b1 + w * 0.2965164f;
            b2 = 0.57000f * b2 + w * 1.0526913f;
            x[i] = b0 + b1 + b2 + w * 0.1848f;
        } else {
            x[i] = w;
        }
    }
    double e = 0.0;
    for (float v : x) e += (double)v * v;
    const float g = rms / (float)std::sqrt(e / (double)n + 1e-30);
    for (float& v : x) v *= g;
    return x;
}

static bool write_wav16(const std::string& path, const std::vector<float>& pcm) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t n = (uint32_t)pcm.size() * 2, sr = 16000, br = sr * 2, fmt = 16, riff = 36 + n;
    const uint16_t pcm_tag = 1, ch = 1, align = 2, bits = 16;
    std::fwrite("RIFF", 1, 4, f); std::fwrite(&riff, 4, 1, f); std::fwrite("WAVEfmt ", 1, 8, f);
    std::fwrite(&fmt, 4, 1, f); std::fwrite(&pcm_tag, 2, 1, f); std::fwrite(&ch, 2, 1, f);
    std::fwrite(&sr, 4, 1, f); std::fwrite(&br, 4, 1, f); std::fwrite(&align, 2, 1, f); std::fwrite(&bits, 2, 1, f);
    std::fwrite("data", 1, 4, f); std::fwrite(&n, 4, 1, f);
    for (float v : pcm) {
        const int16_t s = (int16_t)std::lrintf(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f);
        std::fwrite(&s, 2, 1, f);
    }
    std::fclose(f);
    return true;
}

struct Clip {
    std::vector<float> pcm;
    double speech_a_end = 0, noise_lo = 0, noise_hi = 0, speech_b_start = 0, total = 0;
};

static Clip make_clip(bool pink) {
    Audio a, b;
    CHECK(load_audio_16k_mono("tests/fixtures/two_speakers.wav", a));
    CHECK(load_audio_16k_mono("tests/fixtures/speech.wav", b));
    Clip c;
    c.pcm = a.samples;
    c.speech_a_end = (double)c.pcm.size() / 16000.0;
    c.pcm.insert(c.pcm.end(), 16000, 0.0f);
    const std::vector<float> z = make_noise(pink, 12 * 16000, 0.08f, pink ? 11u : 7u);
    c.noise_lo = (double)c.pcm.size() / 16000.0;
    c.pcm.insert(c.pcm.end(), z.begin(), z.end());
    c.noise_hi = (double)c.pcm.size() / 16000.0;
    c.pcm.insert(c.pcm.end(), 16000, 0.0f);
    c.speech_b_start = (double)c.pcm.size() / 16000.0;
    c.pcm.insert(c.pcm.end(), b.samples.begin(), b.samples.end());
    c.total = (double)c.pcm.size() / 16000.0;
    return c;
}

static std::string vad(parakeet_ctx* ctx, const Clip& c, const char* opts) {
    return take(parakeet_capi_vad_pcm_json(ctx, c.pcm.data(), (int)c.pcm.size(), 16000, opts));
}

// `redux`: the head is the Redux head, so the gate must remove the noise.
static void check_detector(const char* path, const char* what, bool redux, double gate) {
    parakeet_ctx* ctx = parakeet_capi_load(path);
    CHECK(ctx != nullptr);
    if (!ctx) return;
    for (int pink = 0; pink < 2; ++pink) {
        const Clip c = make_clip(pink != 0);
        std::fprintf(stderr, "%s, %s noise\n", what, pink ? "pink" : "white");
        char g[64], gj[128];
        std::snprintf(g, sizeof(g), "%.2f", gate);
        for (const char* mode : {"speech", "segments"}) {
            char o0[96], o1[96], o2[128];
            std::snprintf(o0, sizeof(o0), "{\"mode\":\"%s\"}", mode);
            std::snprintf(o1, sizeof(o1), "{\"mode\":\"%s\",\"run_gate\":0}", mode);
            std::snprintf(gj, sizeof(gj), "{\"mode\":\"%s\",\"run_gate\":%s}", mode, g);
            std::snprintf(o2, sizeof(o2), "{\"mode\":\"%s\",\"run_gate\":%s,\"probabilities\":true}", mode, g);
            const std::string off = vad(ctx, c, o0);
            CHECK(!off.empty());
            CHECK(vad(ctx, c, o1) == off);                          // 0 is byte-identical to no key
            if (std::strcmp(mode, "speech") == 0) CHECK(vad(ctx, c, nullptr) == off);
            const std::string on = vad(ctx, c, gj);
            CHECK(!on.empty());
            const auto s_off = segs_of(off), s_on = segs_of(on);
            const double n_off = overlap(s_off, c.noise_lo + 0.5, c.noise_hi - 0.5);
            const double n_on = overlap(s_on, c.noise_lo + 0.5, c.noise_hi - 0.5);
            std::fprintf(stderr, "  %-8s noise stretch called speech: %.2f s off, %.2f s gate %s; total %.2f -> %.2f s\n",
                         mode, n_off, n_on, g, covered(s_off), covered(s_on));
            // The probabilities in the document are the detector's own, gate or not.
            const std::string withp = vad(ctx, c, o2);
            CHECK(!withp.empty() && withp.find("\"probabilities\"") != std::string::npos);
            if (std::strcmp(mode, "speech") == 0) {
                CHECK(covered(s_on) <= covered(s_off) + 1e-6);      // a gate never adds speech
                CHECK(n_on <= n_off + 1e-6);
                if (redux) {
                    CHECK(n_off > 2.0);                             // the noise does fool the head
                    CHECK(n_on <= 0.25 * n_off);                    // and the gate removes most of it
                    // The speech stretches keep their regions.
                    CHECK(same(within(s_off, 0.0, c.speech_a_end + 0.1), within(s_on, 0.0, c.speech_a_end + 0.1)));
                    CHECK(same(within(s_off, c.speech_b_start - 0.1, c.total), within(s_on, c.speech_b_start - 0.1, c.total)));
                    CHECK(!within(s_on, 0.0, c.speech_a_end + 0.1).empty());
                }
            } else {
                // The cuts follow the segmenter rules (last pause inside the 30 s
                // window), so a long pause can stay inside a piece; the speech
                // itself must be inside the pieces, with the gate on.
                CHECK(!s_on.empty());
                CHECK(overlap(s_on, 0.6, c.speech_a_end - 0.2) > 0.9 * (c.speech_a_end - 0.8));
                CHECK(overlap(s_on, c.speech_b_start + 0.6, c.total - 0.4) > 0.9 * (c.total - c.speech_b_start - 1.0));
            }
        }
        // Bad value through the C-API.
        CHECK(parakeet_capi_vad_pcm_json(ctx, c.pcm.data(), (int)c.pcm.size(), 16000, "{\"run_gate\":1}") == nullptr);
        CHECK(std::string(parakeet_capi_last_error(ctx)).find("run_gate") != std::string::npos);
    }
    parakeet_capi_free(ctx);
}

// A full ASR model with a head: the transcribe paths take the gate.
static void check_transcribe(const char* path, const char* what, double gate) {
    std::fprintf(stderr, "%s: transcribe\n", what);
    std::unique_ptr<Model> m = Model::load(path);
    CHECK(m != nullptr);
    if (!m || !m->config().vad.present) return;
    const Clip c = make_clip(false);
    SegmenterOpts o;
    const Transcription base = m->transcribe_pcm_vad_with_timestamps(c.pcm, 16000, Decoder::kDefault, "", o);
    SegmenterOpts z = o;
    z.run_gate = 0.0f;
    const Transcription zero = m->transcribe_pcm_vad_with_timestamps(c.pcm, 16000, Decoder::kDefault, "", z);
    CHECK(zero.text == base.text && zero.words.size() == base.words.size());
    SegmenterOpts go = o;
    go.run_gate = (float)gate;
    const Transcription on = m->transcribe_pcm_vad_with_timestamps(c.pcm, 16000, Decoder::kDefault, "", go);
    CHECK(!on.words.empty());
    size_t in_noise_on = 0, in_noise_off = 0;
    for (const Word& w : on.words) in_noise_on += (w.start > c.noise_lo + 0.5 && w.start < c.noise_hi - 0.5);
    for (const Word& w : base.words) in_noise_off += (w.start > c.noise_lo + 0.5 && w.start < c.noise_hi - 0.5);
    std::fprintf(stderr, "  words inside the noise stretch: %zu off, %zu gate %.2f; %zu / %zu words in all\n",
                 in_noise_off, in_noise_on, gate, base.words.size(), on.words.size());
    CHECK(in_noise_on == 0);
    CHECK(on.words.size() <= base.words.size());
    CHECK(m->transcribe_pcm_vad(c.pcm, 16000, Decoder::kDefault, "", go) == on.text);

    // C-API: the key reaches the segmenter; 0 and no option give the same document.
    const std::string wav = "test_vad_run_gate_model.tmp.wav";
    CHECK(write_wav16(wav, c.pcm));
    parakeet_ctx* ctx = parakeet_capi_load(path);
    CHECK(ctx != nullptr);
    if (ctx) {
        const std::string a = take(parakeet_capi_transcribe_path_json_vad_with(ctx, nullptr, wav.c_str(), 0, nullptr));
        const std::string b = take(parakeet_capi_transcribe_path_json_vad_with(ctx, nullptr, wav.c_str(), 0, "{\"run_gate\":0}"));
        const std::string d = take(parakeet_capi_transcribe_path_json_vad(ctx, wav.c_str(), 0));
        CHECK(!a.empty() && a == b && a == d);
        char opts[64];
        std::snprintf(opts, sizeof(opts), "{\"run_gate\":%.2f}", gate);
        const std::string e = take(parakeet_capi_transcribe_path_json_vad_with(ctx, nullptr, wav.c_str(), 0, opts));
        CHECK(!e.empty());
        CHECK(parakeet_capi_transcribe_path_json_vad_with(ctx, nullptr, wav.c_str(), 0, "{\"run_gate\":1.5}") == nullptr);
        CHECK(std::string(parakeet_capi_last_error(ctx)).find("run_gate") != std::string::npos);
        // Plain transcribe options have no segmenter: the key is unknown there.
        CHECK(parakeet_capi_transcribe_path_json_with(ctx, wav.c_str(), 0, "{\"run_gate\":0.9}") == nullptr);
        parakeet_capi_free(ctx);
    }
    std::remove(wav.c_str());
}

int main() {
    const char* slice = std::getenv("PARAKEET_TEST_VAD_ONLY_REDUX_GGUF");
    const char* keep = std::getenv("PARAKEET_TEST_GGUF_REDUX_KEEP");
    const char* deq = std::getenv("PARAKEET_TEST_GGUF_REDUX_DEQ");
    const char* ultra = std::getenv("PARAKEET_TEST_GGUF_ULTRA");
    const char* silero = std::getenv("PARAKEET_TEST_SILERO_GGUF");
    if (!slice && !keep && !deq && !ultra && !silero) { std::puts("skip: no model env set"); return 77; }
    if (slice) check_detector(slice, "Redux VAD slice", true, 0.92);
    if (keep) { check_detector(keep, "Redux packed", true, 0.92); check_transcribe(keep, "Redux packed", 0.92); }
    if (deq) check_detector(deq, "Redux dequantized", true, 0.92);
    if (ultra) check_detector(ultra, "Ultra", false, 0.92);
    if (silero) check_detector(silero, "Silero", false, 0.92);
    if (failures) return 1;
    std::puts("test_vad_run_gate_model: OK");
    return 0;
}
