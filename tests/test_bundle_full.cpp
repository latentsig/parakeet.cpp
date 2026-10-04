// A full bundle (asr, diar, ced, voice, vad) against the single-model files it was
// built from: every component gives the same output as its standalone file.
//   ASR transcript, diarization JSON, CED class scores (bitwise), speaker embedding
//   (bitwise) and identity, Silero probabilities (bitwise), named diarization with
//   bundle components. Also: partial loading, selection rules, and the temporary-file
//   fallback of the ced and voice loaders (no file is left behind).
//
// LABEL model; run from the project root (fixtures are relative).
// Env (skip 77 unless all are set):
//   PARAKEET_TEST_BUNDLE_FULL   the bundle (components asr, diar, ced, voice, vad)
//   PARAKEET_TEST_FULL_ASR / _DIAR / _CED / _VOICE / _SILERO   the single-model files
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include "audio_io.hpp"
#include "bundle.hpp"
#include "bundle_extract.hpp"
#include "ced_tagger.hpp"
#include "model.hpp"
#include "parakeet_capi.h"
#include "silero_vad.hpp"
#include "speaker_encoder.hpp"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static unsigned long long bytes_read() {
    std::FILE* f = std::fopen("/proc/self/io", "r");
    if (!f) return 0;
    char k[64];
    unsigned long long v = 0, r = 0;
    while (std::fscanf(f, "%63s %llu", k, &v) == 2)
        if (std::string(k) == "rchar:") r = v;
    std::fclose(f);
    return r;
}
static std::string take(char* p) {
    std::string s = p ? p : "";
    if (p) parakeet_capi_free_string(p);
    return s;
}
static bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return !a.empty() && a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
static int count_entries(const std::string& dir) {
    int n = 0;
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d)) if (e->d_name[0] != '.') ++n;
        closedir(d);
    }
    return n;
}

// Results of the component-by-component comparison, with the loaders told how to open.
struct Results {
    std::string transcript, diar_json, named_json, id;
    std::vector<float> ced, emb, sil;
};

int main() {
    const char* bundle = std::getenv("PARAKEET_TEST_BUNDLE_FULL");
    const char *fa = std::getenv("PARAKEET_TEST_FULL_ASR"), *fd = std::getenv("PARAKEET_TEST_FULL_DIAR"),
               *fc = std::getenv("PARAKEET_TEST_FULL_CED"), *fv = std::getenv("PARAKEET_TEST_FULL_VOICE"),
               *fs = std::getenv("PARAKEET_TEST_FULL_SILERO");
    if (!bundle || !fa || !fd || !fc || !fv || !fs) {
        std::puts("skip: PARAKEET_TEST_BUNDLE_FULL and PARAKEET_TEST_FULL_{ASR,DIAR,CED,VOICE,SILERO} must be set");
        return 77;
    }
    if (!pk::CedTagger::available() || !pk::SpeakerEncoder::available()) {
        std::puts("skip: built without CED or voice-detect");
        return 77;
    }
    const std::string speech = "tests/fixtures/speech.wav", two = "tests/fixtures/two_speakers.wav";
    pk::Audio a_speech, a_two;
    CHECK(pk::load_audio_16k_mono(speech, a_speech) && pk::load_audio_16k_mono(two, a_two));
    struct stat st{};
    CHECK(::stat(bundle, &st) == 0);

    // ---- header: five components, the licences of record ----
    pk::BundleInfo info;
    std::string err;
    CHECK(pk::read_bundle_info(bundle, info, &err));
    const char* want[][3] = {{"asr", "asr", "CC-BY-4.0"}, {"diar", "diar", "OpenMDW-1.1"}, {"ced", "ced", "Apache-2.0"},
                             {"voice", "voice", ""}, {"vad", "vad", "MIT"}};
    for (auto& w : want) {
        const pk::BundleComponent* c = info.find(w[0]);
        CHECK(c && c->kind == w[1] && !c->license.empty() && !c->attribution.empty() && !c->source_sha256.empty());
        if (c && w[2][0]) CHECK(c->license == w[2]);
    }
    const pk::BundleComponent* vc = info.find("voice");
    CHECK(vc && (vc->license == "CC-BY-4.0" || vc->license == "Apache-2.0"));

    // ---- selection ----
    {
        parakeet_ctx* d = parakeet_capi_load(bundle);
        CHECK(d && parakeet_capi_model_kind(d) == PARAKEET_MODEL_KIND_ASR);   // plain load picks the ASR component
        parakeet_capi_free(d);
        struct { const char* n; int kind; } k[] = {{"asr", PARAKEET_MODEL_KIND_ASR}, {"diar", PARAKEET_MODEL_KIND_DIARIZATION},
            {"ced", PARAKEET_MODEL_KIND_SOUND}, {"voice", PARAKEET_MODEL_KIND_SPEAKER}, {"vad", PARAKEET_MODEL_KIND_VAD}};
        for (auto& e : k) {
            parakeet_ctx* c = parakeet_capi_load_component(bundle, e.n);
            CHECK(c && parakeet_capi_model_kind(c) == e.kind);
            parakeet_capi_free(c);
        }
        CHECK(!pk::Model::load(bundle));                                // a bundle needs a component
        CHECK(!pk::Model::load(bundle, "diar"));                        // wrong kind for the ASR loader
        CHECK(!pk::DiarizationModel::load(bundle, "asr"));              // and for the diarization loader
        // The unmodified third-party loaders (what an older parakeet build calls) refuse the bundle.
        CHECK(!pk::CedTagger::load(bundle));
        CHECK(!pk::SpeakerEncoder::load(bundle));
        CHECK(!pk::CedTagger::load(bundle, "voice"));
        CHECK(!pk::SpeakerEncoder::load(bundle, "ced"));
        CHECK(!pk::SpeakerEncoder::load(bundle, "nope"));
        const std::string js = take(parakeet_capi_bundle_components_json(bundle));
        for (auto& w : want) CHECK(js.find(std::string("\"kind\":\"") + w[1] + "\"") != std::string::npos);
    }

    // ---- standalone results ----
    Results solo;
    {
        auto m = pk::Model::load(fa);
        CHECK(m != nullptr);
        if (m) solo.transcript = m->transcribe_path(speech);
        parakeet_ctx* d = parakeet_capi_load(fd);
        CHECK(d != nullptr);
        if (d) solo.diar_json = take(parakeet_capi_diarize_path(d, two.c_str()));
        auto t = pk::CedTagger::load(fc);
        CHECK(t != nullptr);
        if (t) { pk::SoundScorer sc = t->scorer(); CHECK(sc(a_speech.samples.data(), (int)a_speech.samples.size(), solo.ced)); }
        auto e = pk::SpeakerEncoder::load(fv);
        CHECK(e != nullptr);
        if (e) CHECK(e->embed(a_speech.samples.data(), (int)a_speech.samples.size(), solo.emb));
        parakeet_ctx* sp = parakeet_capi_load(fv);
        CHECK(sp != nullptr);
        if (sp && d) {
            parakeet_speaker_registry* reg = parakeet_capi_speaker_registry_new();
            solo.named_json = take(parakeet_capi_diarize_named_pcm_json(d, sp, reg, a_two.samples.data(),
                                   (int)a_two.samples.size(), 16000, 0.5f, 0.05f));
            parakeet_capi_speaker_registry_free(reg);
            solo.id = parakeet_capi_speaker_identity(sp) ? parakeet_capi_speaker_identity(sp) : "";
        }
        std::string se;
        auto s = pk::SileroVad::load(fs, &se);
        CHECK(s != nullptr);
        if (s) solo.sil = s->probabilities(a_speech.samples.data(), a_speech.samples.size(), 16000);
        parakeet_capi_free(d); parakeet_capi_free(sp);
    }
    CHECK(!solo.transcript.empty() && !solo.diar_json.empty() && !solo.ced.empty() && !solo.emb.empty() && !solo.sil.empty());
    CHECK(!solo.named_json.empty() && solo.id.rfind("sha256:", 0) == 0);
    std::printf("transcript: %s\ndiarization: %s\n", solo.transcript.c_str(), solo.diar_json.c_str());

    // ---- the same through the bundle, with the in-memory file and with the temporary-file fallback ----
    char tmpl[] = "/tmp/pk_bundle_full_XXXXXX";
    const std::string tmpdir = ::mkdtemp(tmpl) ? tmpl : "";
    CHECK(!tmpdir.empty());
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            ::setenv("PARAKEET_BUNDLE_NO_MEMFD", "1", 1);
            ::setenv("TMPDIR", tmpdir.c_str(), 1);
        }
        std::printf("-- pass %d (%s)\n", pass, pass ? "temporary file" : "memfd");
        {
            auto cf = pk::ComponentFile::create(bundle, "ced", &err);
            CHECK(cf && cf->in_memory() == (pass == 0));
            if (cf && pass == 1) CHECK(count_entries(tmpdir) == 1);
        }
        CHECK(count_entries(tmpdir) == 0);   // the temporary file is gone once the object is

        auto m = pk::Model::load(bundle, "asr");
        CHECK(m != nullptr);
        if (m) CHECK(m->transcribe_path(speech) == solo.transcript);

        parakeet_ctx* d = parakeet_capi_load_component(bundle, "diar");
        CHECK(d != nullptr);
        if (d) CHECK(take(parakeet_capi_diarize_path(d, two.c_str())) == solo.diar_json);

        const unsigned long long r0 = bytes_read();
        std::string le;
        auto t = pk::CedTagger::load(bundle, "ced", &le);
        const unsigned long long r1 = bytes_read();
        CHECK(t != nullptr);
        const pk::BundleComponent* cc = info.find("ced");
        if (t) {
            std::vector<float> p;
            pk::SoundScorer sc = t->scorer();
            CHECK(sc(a_speech.samples.data(), (int)a_speech.samples.size(), p) && same_bits(p, solo.ced));
            if (r1 > r0 && cc) {
                std::printf("ced: read %.1f MB (component %.1f MB, bundle %.1f MB)\n", (double)(r1 - r0) / 1e6,
                            (double)cc->n_bytes / 1e6, (double)st.st_size / 1e6);
                CHECK(r1 - r0 < 3 * cc->n_bytes + (8u << 20));   // copy into the file, then the loader reads it
                CHECK(r1 - r0 < (unsigned long long)st.st_size / 2);
            }
        }
        const unsigned long long q0 = bytes_read();
        auto e = pk::SpeakerEncoder::load(bundle, "voice", &le);
        const unsigned long long q1 = bytes_read();
        CHECK(e != nullptr);
        if (e) {
            std::vector<float> v;
            CHECK(e->embed(a_speech.samples.data(), (int)a_speech.samples.size(), v) && same_bits(v, solo.emb));
            if (q1 > q0 && vc) CHECK(q1 - q0 < 3 * vc->n_bytes + (8u << 20));
        }
        parakeet_ctx* sp = parakeet_capi_load_component(bundle, "voice");
        CHECK(sp != nullptr);
        if (sp && d) {
            CHECK(std::string(parakeet_capi_speaker_identity(sp)) == solo.id);   // same identity as the standalone file
            parakeet_speaker_registry* reg = parakeet_capi_speaker_registry_new();
            CHECK(take(parakeet_capi_diarize_named_pcm_json(d, sp, reg, a_two.samples.data(), (int)a_two.samples.size(),
                                                            16000, 0.5f, 0.05f)) == solo.named_json);
            parakeet_capi_speaker_registry_free(reg);
        }
        std::string se;
        auto s = pk::SileroVad::load(bundle, &se, "vad");
        CHECK(s != nullptr);
        if (s) CHECK(same_bits(s->probabilities(a_speech.samples.data(), a_speech.samples.size(), 16000), solo.sil));
        // C-API sound context from the bundle
        parakeet_ctx* tg = parakeet_capi_load_component(bundle, "ced");
        CHECK(tg && parakeet_capi_num_classes(tg) > 0 && parakeet_capi_class_label(tg, 0));
        parakeet_capi_free(tg); parakeet_capi_free(d); parakeet_capi_free(sp);
        CHECK(count_entries(tmpdir) == 0);
    }
    ::unsetenv("PARAKEET_BUNDLE_NO_MEMFD");
    ::rmdir(tmpdir.c_str());

    if (failures) return 1;
    std::printf("test_bundle_full OK\n");
    return 0;
}
