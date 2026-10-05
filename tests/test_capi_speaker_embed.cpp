// parakeet_capi_speaker_embed_pcm / parakeet_capi_free_floats.
// The argument checks need no model. The embedding checks run on each of:
//   PARAKEET_TEST_VD_GGUF       a standalone speaker-encoder GGUF (voice-detect.cpp)
//   PARAKEET_TEST_BUNDLE_VOICE  a bundle GGUF with a "voice" component (component name "voice",
//                               or PARAKEET_TEST_BUNDLE_VOICE_COMPONENT)
// With neither set the test skips (77) after the argument checks. With both set, it also checks
// that the bundle component embeds in the same space as the standalone file when it was built
// from the same encoder (PARAKEET_TEST_BUNDLE_VOICE_SAME=1; cosine >= 0.999).
#include "parakeet_capi.h"

#include "audio_io.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__);  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static std::vector<float> slice(const std::vector<float>& x, double a, double b) {
    return std::vector<float>(x.begin() + (long)(a * 16000), x.begin() + (long)(b * 16000));
}

static std::vector<float> embed(parakeet_ctx* c, const std::vector<float>& pcm, int sr) {
    float* e = nullptr;
    int d = 0;
    std::vector<float> v;
    if (parakeet_capi_speaker_embed_pcm(c, pcm.data(), (int)pcm.size(), sr, &e, &d) == 0 && e && d > 0)
        v.assign(e, e + d);
    parakeet_capi_free_floats(e);
    return v;
}

static double dot(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) s += (double)a[i] * b[i];
    return s;
}
static double cosine(const std::vector<float>& a, const std::vector<float>& b) {
    return dot(a, b) / (std::sqrt(dot(a, a)) * std::sqrt(dot(b, b)) + 1e-12);
}

static void run(const char* label, parakeet_ctx* spk, const std::vector<float>& wav) {
    std::fprintf(stderr, "-- %s\n", label);
    CHECK(parakeet_capi_model_kind(spk) == PARAKEET_MODEL_KIND_SPEAKER);
    const int dim = parakeet_capi_speaker_dim(spk);
    CHECK(dim >= 128);

    const auto a0 = slice(wav, 0.6, 4.6), a1 = slice(wav, 14.9, 18.7), b0 = slice(wav, 6.9, 10.9);
    const auto ea0 = embed(spk, a0, 16000);
    CHECK((int)ea0.size() == dim);
    CHECK(std::fabs(std::sqrt(dot(ea0, ea0)) - 1.0) < 1e-3);  // L2-normalized

    // The out parameters on success.
    {
        float* e = nullptr;
        int d = -1;
        CHECK(parakeet_capi_speaker_embed_pcm(spk, a0.data(), (int)a0.size(), 16000, &e, &d) == 0);
        CHECK(e != nullptr && d == dim);
        // Determinism: the same audio twice gives the same vector.
        CHECK(e && d == dim && std::memcmp(e, ea0.data(), (size_t)dim * sizeof(float)) == 0);
        parakeet_capi_free_floats(e);
    }

    // Same voice scores higher than two different voices.
    const auto ea1 = embed(spk, a1, 16000), eb0 = embed(spk, b0, 16000);
    const double same = cosine(ea0, ea1), diff = cosine(ea0, eb0);
    std::fprintf(stderr, "   same voice %.3f, different voices %.3f\n", same, diff);
    CHECK(same > diff);
    CHECK(same > 0.5);

    // Two different synthetic signals (a tone and a noise burst) give valid, different vectors.
    // Their cosine is not compared with the voice scores: such signals are outside the
    // training distribution.
    {
        std::vector<float> tone(32000), noise(32000);
        unsigned s = 12345;
        for (size_t i = 0; i < tone.size(); ++i) {
            tone[i] = 0.3f * std::sin(2.0f * 3.14159265f * 220.0f * (float)i / 16000.0f);
            s = s * 1664525u + 1013904223u;
            noise[i] = 0.3f * ((float)(s >> 8) / 8388608.0f - 1.0f);
        }
        const auto et = embed(spk, tone, 16000), en = embed(spk, noise, 16000);
        CHECK((int)et.size() == dim && (int)en.size() == dim);
        CHECK(et != en && cosine(et, en) < 0.999);
        CHECK(embed(spk, tone, 16000) == et);
    }

    // Another sample rate is resampled: 8 kHz made by averaging pairs of samples.
    {
        std::vector<float> lo(a0.size() / 2);
        for (size_t i = 0; i < lo.size(); ++i) lo[i] = 0.5f * (a0[2 * i] + a0[2 * i + 1]);
        const auto e8 = embed(spk, lo, 8000);
        CHECK((int)e8.size() == dim);
        CHECK(cosine(e8, ea0) > 0.7);
        CHECK(!embed(spk, a0, 44100).empty());
    }

    // Errors set the outputs to NULL / 0 and the message on the ctx.
    float* e = reinterpret_cast<float*>(0x1);
    int d = 7;
    CHECK(parakeet_capi_speaker_embed_pcm(spk, a0.data(), (int)a0.size(), 0, &e, &d) != 0);
    CHECK(e == nullptr && d == 0);
    CHECK(std::strlen(parakeet_capi_last_error(spk)) > 0);
    e = reinterpret_cast<float*>(0x1);
    CHECK(parakeet_capi_speaker_embed_pcm(spk, a0.data(), (int)a0.size(), -16000, &e, &d) != 0 && e == nullptr);
    CHECK(parakeet_capi_speaker_embed_pcm(spk, nullptr, 100, 16000, &e, &d) != 0 && e == nullptr);
    CHECK(parakeet_capi_speaker_embed_pcm(spk, a0.data(), 0, 16000, &e, &d) != 0 && e == nullptr);
    CHECK(parakeet_capi_speaker_embed_pcm(spk, a0.data(), -5, 16000, &e, &d) != 0 && e == nullptr);
    CHECK(parakeet_capi_speaker_embed_pcm(spk, a0.data(), (int)a0.size(), 16000, nullptr, &d) != 0);
    CHECK(parakeet_capi_speaker_embed_pcm(spk, a0.data(), (int)a0.size(), 16000, &e, nullptr) != 0);
    // The ctx still works after errors.
    CHECK(embed(spk, a0, 16000) == ea0);
}

int main() {
    // Argument checks that need no model.
    {
        float* e = reinterpret_cast<float*>(0x1);
        int d = 7;
        const float pcm[160] = {0};
        CHECK(parakeet_capi_speaker_embed_pcm(nullptr, pcm, 160, 16000, &e, &d) != 0);
        CHECK(e == nullptr && d == 0);
        parakeet_capi_free_floats(nullptr);  // safe on NULL
    }

    const char* vd = std::getenv("PARAKEET_TEST_VD_GGUF");
    const char* bundle = std::getenv("PARAKEET_TEST_BUNDLE_VOICE");
    if (!vd && !bundle) {
        std::puts("skip: PARAKEET_TEST_VD_GGUF or PARAKEET_TEST_BUNDLE_VOICE must be set");
        return failures ? 1 : 77;
    }

    pk::Audio wav;
    if (!pk::load_audio_16k_mono(std::string(PK_SOURCE_DIR) + "/tests/fixtures/two_speakers.wav", wav)) return 1;

    parakeet_ctx *s1 = nullptr, *s2 = nullptr;
    if (vd) {
        s1 = parakeet_capi_load(vd);
        if (!s1) { std::fprintf(stderr, "skip: load speaker model (built without voice-detect?)\n"); return 77; }
        run("standalone speaker GGUF", s1, wav.samples);
    }
    if (bundle) {
        const char* comp = std::getenv("PARAKEET_TEST_BUNDLE_VOICE_COMPONENT");
        s2 = parakeet_capi_load_component(bundle, comp ? comp : "voice");
        if (!s2) {
            std::fprintf(stderr, "FAIL: load bundle voice component: %s\n", parakeet_capi_load_error());
            ++failures;
        } else {
            run("bundle voice component", s2, wav.samples);
            // A context of another kind is refused with a message on that ctx.
            parakeet_ctx* other = parakeet_capi_load_component(bundle, "asr");
            if (other) {
                float* e = reinterpret_cast<float*>(0x1);
                int d = 7;
                const float pcm[1600] = {0};
                CHECK(parakeet_capi_speaker_embed_pcm(other, pcm, 1600, 16000, &e, &d) != 0);
                CHECK(e == nullptr && d == 0 && std::strlen(parakeet_capi_last_error(other)) > 0);
                parakeet_capi_free(other);
            }
        }
    }
    if (s1 && s2) {
        const char* same = std::getenv("PARAKEET_TEST_BUNDLE_VOICE_SAME");
        if (same && same[0] == '1') {
            const auto a0 = slice(wav.samples, 0.6, 4.6);
            const auto x = embed(s1, a0, 16000), y = embed(s2, a0, 16000);
            const double c = cosine(x, y);
            std::fprintf(stderr, "   standalone vs bundle cosine %.6f\n", c);
            CHECK(x.size() == y.size() && c >= 0.999);
        }
    }
    parakeet_capi_free(s1);
    parakeet_capi_free(s2);
    std::fprintf(stderr, failures ? "FAILED (%d)\n" : "ok\n", failures);
    return failures ? 1 : 0;
}
