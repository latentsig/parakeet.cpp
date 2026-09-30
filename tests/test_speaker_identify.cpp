// Speaker identification end to end: real diarization + real speaker encoder.
//
//   PARAKEET_TEST_DIAR_GGUF   diarization GGUF   (required, else skip 77)
//   PARAKEET_TEST_VD_GGUF     speaker encoder GGUF (required, else skip 77)
//
// Enrolls the two voices of tests/fixtures/two_speakers.wav (LibriSpeech 1272 = A,
// 2086 = B) from clips that do not overlap the segments being checked, then streams
// the whole file and checks the diarization slots get the right names. NeMo's
// segments for the fixture: A 0.50-5.52 and 14.78-18.75, B 6.85-13.49 and 20.10-23.60.
#include "audio_io.hpp"
#include "diarization.hpp"
#include "scene_stream.hpp"
#include "speaker_encoder.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
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

static std::vector<float> slice(const std::vector<float>& x, double a, double b) {
    return std::vector<float>(x.begin() + (long)(a * 16000), x.begin() + (long)(b * 16000));
}

int main() {
    const char* diar_path = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    const char* vd_path = std::getenv("PARAKEET_TEST_VD_GGUF");
    if (!diar_path || !vd_path) return 77;
    if (!SpeakerEncoder::available()) return 77;

    auto diar = DiarizationModel::load(diar_path);
    auto enc = SpeakerEncoder::load(vd_path);
    if (!diar || !enc) { std::fprintf(stderr, "FAIL: load models\n"); return 1; }

    Audio wav;
    if (!load_audio_16k_mono(std::string(PK_SOURCE_DIR) + "/tests/fixtures/two_speakers.wav", wav)) {
        std::fprintf(stderr, "FAIL: load wav\n");
        return 1;
    }

    // Enroll from the first turn of A and the first turn of B only (about 4 s each).
    SpeakerRegistry reg;
    std::vector<float> e;
    auto a0 = slice(wav.samples, 0.6, 4.6);
    auto b0 = slice(wav.samples, 6.9, 10.9);
    CHECK(enc->embed(a0.data(), (int)a0.size(), e)); reg.enroll("speaker_a", e);
    CHECK(enc->embed(b0.data(), (int)b0.size(), e)); reg.enroll("speaker_b", e);

    SceneParts parts;
    parts.diar = diar.get();
    parts.speaker_embed = enc->embedder();
    parts.registry = &reg;
    SceneStream stream(parts);

    std::map<int, std::string> final_names;
    const int chunk = 3200;   // 200 ms
    const int n = (int)wav.samples.size();
    for (int lo = 0; lo < n; lo += chunk) {
        const int len = std::min(chunk, n - lo);
        const SceneUpdate u = stream.feed(wav.samples.data() + lo, len, lo + len >= n);
        for (const auto& kv : u.names) final_names[kv.first] = kv.second.name;
    }

    // Slot numbers are arrival order, so slot 0 is voice A here (A speaks first).
    CHECK(final_names.size() == 2);
    CHECK(final_names[0] == "speaker_a");
    CHECK(final_names[1] == "speaker_b");

    // Speaker part without diarization, or without a registry, is rejected clearly.
    {
        SceneParts no_diar;
        no_diar.speaker_embed = enc->embedder();
        no_diar.registry = &reg;
        bool threw = false;
        try { SceneStream s(no_diar); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        SceneParts no_reg;
        no_reg.diar = diar.get();
        no_reg.speaker_embed = enc->embedder();
        threw = false;
        try { SceneStream s(no_reg); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }

    // With no speaker part the update carries no names (existing behavior).
    {
        SceneParts plain;
        plain.diar = diar.get();
        SceneStream s(plain);
        const SceneUpdate u = s.feed(wav.samples.data(), (int)wav.samples.size(), true);
        CHECK(u.names.empty());
    }

    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_speaker_identify: PASS\n");
    return 0;
}
