// pk::SpeakerEncoder against a real voice-detect GGUF.
//
//   PARAKEET_TEST_VD_GGUF      speaker encoder GGUF (required, else skip 77)
//   PARAKEET_TEST_VD_REF_WAV   optional: a WAV whose reference embedding is in ...
//   PARAKEET_TEST_VD_REF_JSON  ... this file, the output of
//                              `voicedetect-cli embed --model <gguf> --input <wav> --json`
//                              from a standalone voice-detect.cpp build. When both are
//                              set the folded encoder must match it (cosine >= 0.9999).
//
// The functional check uses tests/fixtures/two_speakers.wav (LibriSpeech 1272 and
// 2086, A-B-A-B). NeMo's segments for it: A 0.50-5.52 and 14.78-18.75, B 6.85-10.82
// and 20.10-23.60. Two clips of the same voice must score higher than two clips of
// different voices.
#include "audio_io.hpp"
#include "speaker_encoder.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace pk;

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__);  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) { d += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return d / std::sqrt(na * nb);
}

static std::vector<float> slice(const std::vector<float>& x, double a, double b) {
    return std::vector<float>(x.begin() + (long)(a * 16000), x.begin() + (long)(b * 16000));
}

// Reads "embedding":[...] from voicedetect-cli --json output.
static std::vector<float> read_ref_json(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string s = ss.str();
    const size_t p = s.find("\"embedding\":[");
    std::vector<float> v;
    if (p == std::string::npos) return v;
    const char* c = s.c_str() + p + std::strlen("\"embedding\":[");
    char* end = nullptr;
    while (*c && *c != ']') {
        v.push_back(std::strtof(c, &end));
        c = end;
        if (*c == ',') ++c;
    }
    return v;
}

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_VD_GGUF");
    if (!gguf) return 77;
    if (!SpeakerEncoder::available()) { std::printf("built without PARAKEET_WITH_VOICEDETECT\n"); return 77; }
    CHECK(gguf_is_voicedetect(gguf));
    CHECK(!gguf_is_voicedetect("/nonexistent.gguf"));

    auto enc = SpeakerEncoder::load(gguf);
    if (!enc) { std::fprintf(stderr, "FAIL: load %s\n", gguf); return 1; }
    CHECK(enc->dim() >= 128 && enc->dim() <= 1024);

    Audio wav;
    if (!load_audio_16k_mono(std::string(PK_SOURCE_DIR) + "/tests/fixtures/two_speakers.wav", wav)) {
        std::fprintf(stderr, "FAIL: load two_speakers.wav\n");
        return 1;
    }
    auto emb = [&](double a, double b) {
        std::vector<float> e;
        const auto pcm = slice(wav.samples, a, b);
        if (!enc->embed(pcm.data(), (int)pcm.size(), e)) { std::fprintf(stderr, "embed failed: %s\n", enc->last_error().c_str()); ++failures; }
        return e;
    };
    const auto a1 = emb(0.6, 5.4), a2 = emb(14.9, 18.7), b1 = emb(6.9, 10.7), b2 = emb(20.2, 23.5);
    CHECK((int)a1.size() == enc->dim());
    double n2 = 0;
    for (float x : a1) n2 += (double)x * x;
    CHECK(std::fabs(n2 - 1.0) < 1e-3);                       // L2-normalized
    const double same = 0.5 * (cosine(a1, a2) + cosine(b1, b2));
    const double diff = 0.25 * (cosine(a1, b1) + cosine(a1, b2) + cosine(a2, b1) + cosine(a2, b2));
    std::printf("same-speaker cosine %.3f, different-speaker cosine %.3f\n", same, diff);
    CHECK(same > diff + 0.1);

    // Empty and tiny inputs fail cleanly instead of crashing.
    std::vector<float> e;
    CHECK(!enc->embed(nullptr, 0, e));
    CHECK(!enc->last_error().empty());

    const char* ref_wav = std::getenv("PARAKEET_TEST_VD_REF_WAV");
    const char* ref_json = std::getenv("PARAKEET_TEST_VD_REF_JSON");
    if (ref_wav && ref_json) {
        Audio r;
        CHECK(load_audio_16k_mono(ref_wav, r));
        std::vector<float> got;
        CHECK(enc->embed(r.samples.data(), (int)r.samples.size(), got));
        const auto want = read_ref_json(ref_json);
        CHECK(want.size() == got.size());
        const double c = want.size() == got.size() ? cosine(got, want) : 0.0;
        std::printf("folded vs standalone cosine %.6f\n", c);
        CHECK(c >= 0.9999);
    }

    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_speaker_encoder: PASS\n");
    return 0;
}
