// Speaker identification end to end: real diarization + real speaker encoder.
//
//   PARAKEET_TEST_DIAR_GGUF   diarization GGUF   (required, else skip 77)
//   PARAKEET_TEST_VD_GGUF     speaker encoder GGUF (required, else skip 77)
//   PARAKEET_TEST_GGUF        ASR GGUF (optional: enables the named-utterance block)
//
// Enrolls the two voices of tests/fixtures/two_speakers.wav (LibriSpeech 1272 = A,
// 2086 = B) from their first turns, then streams the whole file and checks the
// diarization slots get the right names. The enrollment clips come from inside the
// streamed segments (same recording), so on its own this is parity-style evidence.
// What makes it discriminating: the voices are enrolled in the reverse of their
// arrival order (so slot i cannot map to registry entry i), and a registry that
// lacks voice A must leave slot 0 unnamed. NeMo's segments for the fixture: A
// 0.50-5.52 and 14.78-18.75, B 6.85-13.49 and 20.10-23.60.
#include "audio_io.hpp"
#include "diarization.hpp"
#include "model.hpp"
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

    auto a0 = slice(wav.samples, 0.6, 4.6);
    auto b0 = slice(wav.samples, 6.9, 10.9);
    std::vector<float> e;

    // Streams the file and returns each slot's final name.
    auto run = [&](const SpeakerRegistry& reg, const SpeakerIdOpts& opts = SpeakerIdOpts()) {
        SceneParts parts;
        parts.diar = diar.get();
        parts.speaker_embed = enc->embedder();
        parts.registry = &reg;
        parts.speaker_opts = opts;
        SceneStream stream(parts);
        std::map<int, std::string> names;
        const int chunk = 3200;   // 200 ms
        const int n = (int)wav.samples.size();
        for (int lo = 0; lo < n; lo += chunk) {
            const int len = std::min(chunk, n - lo);
            const SceneUpdate u = stream.feed(wav.samples.data() + lo, len, lo + len >= n);
            for (const auto& kv : u.names) {
                names[kv.first] = kv.second.name;
                if (std::getenv("PK_TEST_DEBUG"))
                    std::fprintf(stderr, "DBG t=%.1f slot%d '%s' %.3f\n", u.t, kv.first,
                                 kv.second.name.c_str(), kv.second.score);
            }
        }
        return names;
    };

    // Enroll in reverse arrival order: B's clip first, A's second.
    {
        SpeakerRegistry reg;
        CHECK(enc->embed(b0.data(), (int)b0.size(), e)); reg.enroll("second_voice", e);
        CHECK(enc->embed(a0.data(), (int)a0.size(), e)); reg.enroll("first_voice", e);
        auto names = run(reg);
        // Slot numbers are arrival order: slot 0 is voice A (speaks first), slot 1 is B.
        CHECK(names.size() == 2);
        CHECK(names[0] == "first_voice");
        CHECK(names[1] == "second_voice");
    }
    // Only B enrolled: slot 0 (voice A) must stay unnamed, never get B's name.
    {
        SpeakerRegistry reg;
        CHECK(enc->embed(b0.data(), (int)b0.size(), e)); reg.enroll("second_voice", e);
        // ECAPA scored an impostor voice at cosine 0.566 on this fixture, so the
        // acceptance threshold is encoder specific and the default is a starting
        // point (docs/speaker.md carries the per-encoder numbers).
        SpeakerIdOpts strict;
        strict.accept_threshold = 0.7f;
        auto names = run(reg, strict);
        CHECK(names.count(0) == 1 && names[0].empty());
        CHECK(names[1] == "second_voice");
    }

    SpeakerRegistry reg;
    CHECK(enc->embed(a0.data(), (int)a0.size(), e)); reg.enroll("first_voice", e);

    // Bad speaker configurations are rejected with the specific message.
    auto expect_throw = [&](const SceneParts& p, const char* what) {
        try {
            SceneStream s(p);
        } catch (const std::invalid_argument& ex) {
            if (std::string(ex.what()).find(what) != std::string::npos) return;
            std::fprintf(stderr, "FAIL: wrong message '%s', wanted '%s'\n", ex.what(), what);
            ++failures;
            return;
        }
        std::fprintf(stderr, "FAIL: no throw, wanted '%s'\n", what);
        ++failures;
    };
    {
        SceneParts no_diar;
        no_diar.speaker_embed = enc->embedder();
        no_diar.registry = &reg;
        expect_throw(no_diar, "needs a diarization model");
        SceneParts no_reg;
        no_reg.diar = diar.get();
        no_reg.speaker_embed = enc->embedder();
        expect_throw(no_reg, "needs a registry");
        SceneParts bad_opts;
        bad_opts.diar = diar.get();
        bad_opts.speaker_embed = enc->embedder();
        bad_opts.registry = &reg;
        bad_opts.speaker_opts.min_voice_sec = 0;
        expect_throw(bad_opts, "invalid speaker options");
    }

    // With no speaker part the update carries no names (existing behavior).
    {
        SceneParts plain;
        plain.diar = diar.get();
        SceneStream s(plain);
        const SceneUpdate u = s.feed(wav.samples.data(), (int)wav.samples.size(), true);
        CHECK(u.names.empty());
    }

    // ASR + diarization + speaker: the utterances themselves carry names. Words
    // committed before a slot is identified keep their earlier (empty) name, so an
    // empty name is allowed; the other voice's name never is.
    if (const char* asr_path = std::getenv("PARAKEET_TEST_GGUF")) {
        auto asr = Model::load(asr_path);
        if (!asr) { std::fprintf(stderr, "FAIL: load asr\n"); return 1; }
        SpeakerRegistry areg;
        CHECK(enc->embed(b0.data(), (int)b0.size(), e)); areg.enroll("second_voice", e);
        CHECK(enc->embed(a0.data(), (int)a0.size(), e)); areg.enroll("first_voice", e);
        SceneParts parts;
        parts.asr = asr.get();
        parts.diar = diar.get();
        parts.speaker_embed = enc->embedder();
        parts.registry = &areg;
        SceneStream stream(parts);
        std::vector<SpeakerUtterance> utts;
        const int chunk = 3200;
        const int n = (int)wav.samples.size();
        for (int lo = 0; lo < n; lo += chunk) {
            const int len = std::min(chunk, n - lo);
            const SceneUpdate u = stream.feed(wav.samples.data() + lo, len, lo + len >= n);
            utts.insert(utts.end(), u.utterances.begin(), u.utterances.end());
        }
        CHECK(!utts.empty());
        int named0 = 0, named1 = 0, wrong = 0;
        for (const auto& u : utts) {
            if (std::getenv("PK_TEST_DEBUG"))
                std::fprintf(stderr, "UTT slot%d '%s' start=%.2f '%s'\n", u.speaker, u.name.c_str(),
                             u.start, u.text.c_str());
            if (u.speaker == 0 && u.name == "first_voice") ++named0;
            if (u.speaker == 1 && u.name == "second_voice") ++named1;
            if ((u.speaker == 0 && u.name == "second_voice") ||
                (u.speaker == 1 && u.name == "first_voice"))
                ++wrong;
        }
        CHECK(named0 >= 1);
        CHECK(named1 >= 1);
        CHECK(wrong == 0);
    }

    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_speaker_identify: PASS\n");
    return 0;
}
