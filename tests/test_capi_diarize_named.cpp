// Diarize-only naming through the C-API, with a registry filled by
// add_embedding from embeddings the folded encoder computes.
//   PARAKEET_TEST_DIAR_GGUF, PARAKEET_TEST_VD_GGUF (required, else skip 77)
#include "audio_io.hpp"
#include "parakeet_capi.h"
#include "speaker_encoder.hpp"

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

// Splits the "names" object of a document into its names and scores, in order.
static void parse_names(const std::string& doc, std::vector<std::string>& names, std::vector<float>& scores) {
    size_t p = doc.find("\"names\":");
    if (p == std::string::npos) return;
    const std::string key = "\"name\":\"";
    while ((p = doc.find(key, p)) != std::string::npos) {
        p += key.size();
        const size_t e = doc.find('"', p);
        names.push_back(doc.substr(p, e - p));
        const size_t sp = doc.find("\"score\":", e);
        scores.push_back(std::strtof(doc.c_str() + sp + 8, nullptr));
        p = sp;
    }
}

int main() {
    const char* diar_path = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    const char* vd_path = std::getenv("PARAKEET_TEST_VD_GGUF");
    if (!diar_path || !vd_path) return 77;
    if (!pk::SpeakerEncoder::available()) return 77;

    pk::Audio wav;
    if (!pk::load_audio_16k_mono(std::string(PK_SOURCE_DIR) + "/tests/fixtures/two_speakers.wav", wav)) return 1;

    // Embeddings of the two voices, computed outside the C-API on purpose:
    // this is what LocalAI hands over (vectors, not audio).
    auto enc = pk::SpeakerEncoder::load(vd_path);
    if (!enc) { std::fprintf(stderr, "FAIL: load speaker model\n"); return 1; }
    std::vector<float> ea, eb;
    const auto ca = slice(wav.samples, 0.6, 4.6), cb = slice(wav.samples, 6.9, 10.9);
    CHECK(enc->embed(ca.data(), (int)ca.size(), ea));
    CHECK(enc->embed(cb.data(), (int)cb.size(), eb));

    parakeet_ctx* diar = parakeet_capi_load(diar_path);
    parakeet_ctx* spk = parakeet_capi_load(vd_path);
    CHECK(diar && spk);
    parakeet_speaker_registry* reg = parakeet_capi_speaker_registry_new();
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "second_voice", eb.data(), (int)eb.size()) == 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "first_voice", ea.data(), (int)ea.size()) == 0);

    // Defaults: slot 0 is voice A (first to speak), slot 1 is voice B. The registry order is
    // reversed on purpose, so arrival-order naming would fail.
    std::vector<std::string> n1;
    std::vector<float> s1;
    char* j = parakeet_capi_diarize_named_pcm_json(diar, spk, reg, wav.samples.data(), (int)wav.samples.size(),
                                                    16000, 0.0f, 0.0f);
    CHECK(j != nullptr);
    if (j) {
        const std::string s = j;
        CHECK(s.find("\"segments\":[{\"speaker\":0") != std::string::npos);
        CHECK(s.find("\"names\":{\"0\":{\"name\":\"first_voice\"") != std::string::npos);
        CHECK(s.find("\"1\":{\"name\":\"second_voice\"") != std::string::npos);
        parse_names(s, n1, s1);
        parakeet_capi_free_string(j);
    }

    // A registry built with enroll from the same two clips gives the same names and scores.
    {
        parakeet_speaker_registry* reg2 = parakeet_capi_speaker_registry_new();
        CHECK(parakeet_capi_speaker_enroll(reg2, spk, "first_voice", ca.data(), (int)ca.size(), 16000) == 0);
        CHECK(parakeet_capi_speaker_enroll(reg2, spk, "second_voice", cb.data(), (int)cb.size(), 16000) == 0);
        std::vector<std::string> n2;
        std::vector<float> s2;
        j = parakeet_capi_diarize_named_pcm_json(diar, spk, reg2, wav.samples.data(), (int)wav.samples.size(),
                                                  16000, 0.0f, 0.0f);
        CHECK(j != nullptr);
        if (j) {
            parse_names(j, n2, s2);
            parakeet_capi_free_string(j);
        }
        CHECK(!n1.empty() && n1 == n2 && s1.size() == s2.size());
        for (size_t i = 0; i < s1.size() && i < s2.size(); ++i) CHECK(std::fabs(s1[i] - s2[i]) < 1e-3f);
        parakeet_capi_speaker_registry_free(reg2);
    }

    // The threshold reaches the identifier: above every genuine score, nothing is named.
    j = parakeet_capi_diarize_named_pcm_json(diar, spk, reg, wav.samples.data(), (int)wav.samples.size(),
                                              16000, 0.999f, 0.0f);
    CHECK(j != nullptr);
    if (j) {
        const std::string s = j;
        CHECK(s.find("\"name\":\"\"") != std::string::npos);
        CHECK(s.find("first_voice") == std::string::npos && s.find("second_voice") == std::string::npos);
        parakeet_capi_free_string(j);
    }

    // Invalid options and a non-16k input rate.
    CHECK(parakeet_capi_diarize_named_pcm_json(diar, spk, reg, wav.samples.data(), (int)wav.samples.size(),
                                                16000, 2.0f, 0.0f) == nullptr);
    CHECK(std::strstr(parakeet_capi_last_error(spk), "invalid speaker options") != nullptr);
    {
        std::vector<float> twice;   // the same audio at 32 kHz (each sample repeated): must still name both
        for (float v : wav.samples) { twice.push_back(v); twice.push_back(v); }
        j = parakeet_capi_diarize_named_pcm_json(diar, spk, reg, twice.data(), (int)twice.size(), 32000, 0.0f, 0.0f);
        CHECK(j != nullptr);
        if (j) {
            const std::string s = j;
            CHECK(s.find("first_voice") != std::string::npos && s.find("second_voice") != std::string::npos);
            parakeet_capi_free_string(j);
        }
    }

    // Error paths.
    CHECK(parakeet_capi_diarize_named_pcm_json(diar, nullptr, reg, wav.samples.data(), 16000, 16000, 0, 0) == nullptr);
    CHECK(parakeet_capi_diarize_named_pcm_json(diar, spk, nullptr, wav.samples.data(), 16000, 16000, 0, 0) == nullptr);
    CHECK(std::strstr(parakeet_capi_last_error(spk), "needs a registry") != nullptr);
    CHECK(parakeet_capi_diarize_named_pcm_json(spk, spk, reg, wav.samples.data(), 16000, 16000, 0, 0) == nullptr);
    CHECK(parakeet_capi_diarize_named_pcm_json(diar, diar, reg, wav.samples.data(), 16000, 16000, 0, 0) == nullptr);
    CHECK(parakeet_capi_diarize_named_pcm_json(diar, spk, reg, nullptr, 16000, 16000, 0, 0) == nullptr);
    CHECK(parakeet_capi_diarize_named_pcm_json(diar, spk, reg, wav.samples.data(), 0, 16000, 0, 0) == nullptr);
    // A registry made for a different encoder size is refused with both sizes in the message.
    {
        parakeet_speaker_registry* wrong = parakeet_capi_speaker_registry_new();
        const float small[3] = {1, 0, 0};
        CHECK(parakeet_capi_speaker_registry_add_embedding(wrong, "x", small, 3) == 0);
        CHECK(parakeet_capi_diarize_named_pcm_json(diar, spk, wrong, wav.samples.data(), 16000, 16000, 0, 0) == nullptr);
        CHECK(std::strstr(parakeet_capi_last_error(spk), "embeddings") != nullptr);
        parakeet_capi_speaker_registry_free(wrong);
    }

    parakeet_capi_speaker_registry_free(reg);
    parakeet_capi_free(diar);
    parakeet_capi_free(spk);
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_capi_diarize_named: PASS\n");
    return 0;
}
