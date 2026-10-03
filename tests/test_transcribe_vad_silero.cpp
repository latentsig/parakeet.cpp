// Segmented transcription with Silero VAD on an ASR model that has no VAD head:
// a long synthetic clip gives the plain transcript to within the usual
// tolerance, short audio is unchanged, timestamps stay monotonic, and the
// C-API function parakeet_capi_transcribe_path_json_vad_with agrees with the
// C++ API.
//
// Env: PARAKEET_TEST_SILERO_GGUF (Silero VAD) and PARAKEET_TEST_GGUF (any ASR
// model, for example the v3 TDT model). Skips (77) without them. Run from the
// project root.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "audio_io.hpp"
#include "model.hpp"
#include "parakeet_capi.h"
#include "silero_vad.hpp"

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
    const char* silero_path = std::getenv("PARAKEET_TEST_SILERO_GGUF");
    const char* asr_path = std::getenv("PARAKEET_TEST_GGUF");
    if (!silero_path || !asr_path) { std::puts("skip: PARAKEET_TEST_SILERO_GGUF or PARAKEET_TEST_GGUF unset"); return 77; }

    std::string err;
    std::unique_ptr<SileroVad> sv = SileroVad::load(silero_path, &err);
    if (!sv) { std::fprintf(stderr, "silero load: %s\n", err.c_str()); return 1; }
    std::unique_ptr<Model> m = Model::load(asr_path);
    if (!m) { std::fprintf(stderr, "asr load failed\n"); return 1; }
    const SileroVad* svp = sv.get();
    const Model::VadProbabilityFn fn = [svp](const std::vector<float>& pcm) {
        return svp->probabilities(pcm.data(), pcm.size(), 16000);
    };
    SegmenterOpts opts = default_segmenter_opts(VadKind::kSilero);

    // Without the external detector a model with no head still reports the old error.
    if (!m->config().vad.present) {
        bool threw = false;
        try { m->transcribe_pcm_vad(std::vector<float>(16000, 0.f), 16000); } catch (const std::exception& e) {
            threw = std::string(e.what()) == "model has no VAD head";
        }
        CHECK(threw);
    }

    // Short audio (<= 30 s) takes the plain path: identical text.
    Audio shortclip;
    CHECK(load_audio_16k_mono("tests/fixtures/speech.wav", shortclip));
    CHECK(m->transcribe_pcm(shortclip.samples, 16000, Decoder::kDefault) ==
          m->transcribe_pcm_vad(shortclip.samples, 16000, Decoder::kDefault, "", opts, &fn));

    // Long synthetic clip (~65 s): fixtures joined by 0.7 s of near silence.
    Audio clip;
    {
        std::mt19937 rng(3);
        std::normal_distribution<float> nd(0.f, 0.0005f);
        const char* parts[] = {"tests/fixtures/two_speakers.wav", "tests/fixtures/speech.wav",
                               "tests/fixtures/two_speakers.wav", "tests/fixtures/speech.wav"};
        for (const char* f : parts) {
            Audio x;
            CHECK(load_audio_16k_mono(f, x));
            clip.samples.insert(clip.samples.end(), x.samples.begin(), x.samples.end());
            for (int i = 0; i < 11200; ++i) clip.samples.push_back(nd(rng));
        }
        clip.sample_rate = 16000;
    }
    const double total = (double)clip.samples.size() / 16000.0;
    {
        const auto segs = segment_by_vad(fn(clip.samples), total, opts);
        std::printf("clip %.2f s: %zu segments:", total, segs.size());
        for (const auto& s : segs) std::printf(" [%.2f-%.2f]", s.start, s.end);
        std::printf("\n");
        CHECK(segs.size() >= 2);
        for (const auto& s : segs) CHECK(s.end - s.start <= opts.max_seg_sec + 1e-6);
    }
    const std::string full = m->transcribe_pcm(clip.samples, 16000, Decoder::kDefault);
    const std::string seg = m->transcribe_pcm_vad(clip.samples, 16000, Decoder::kDefault, "", opts, &fn);
    const auto wf = words_of(full), ws = words_of(seg);
    const double diff = (double)edit_distance(wf, ws) / (double)std::max<size_t>(1, wf.size());
    std::printf("%zu plain words vs %zu segmented, word diff %.3f\n", wf.size(), ws.size(), diff);
    CHECK(!ws.empty());
    CHECK(diff < 0.15);

    const Transcription tr = m->transcribe_pcm_vad_with_timestamps(clip.samples, 16000, Decoder::kDefault, "", opts, &fn);
    CHECK(!tr.words.empty() && words_of(tr.text) == ws);
    float prev = -1.0f;
    for (const Word& w : tr.words) {
        CHECK(w.start >= prev - 1e-3f && w.end >= w.start);
        prev = w.start;
    }
    if (!tr.words.empty()) CHECK(tr.words.back().end <= (float)total + 0.5f);
    for (size_t i = 1; i < tr.tokens.size(); ++i) CHECK(tr.tokens[i].frame >= tr.tokens[i - 1].frame);

    // Audio with no speech gives an empty transcript.
    {
        std::mt19937 rng(5);
        std::normal_distribution<float> nd(0.f, 0.0003f);
        std::vector<float> quiet((size_t)(40 * 16000));
        for (float& x : quiet) x = nd(rng);
        CHECK(m->transcribe_pcm_vad(quiet, 16000, Decoder::kDefault, "", opts, &fn).empty());
    }

    // C-API: same transcript as the C++ call; the options object is honoured.
    {
        const char* wav = "test_transcribe_vad_silero.tmp.wav";
        // Write the clip as a 16-bit mono WAV.
        {
            std::ofstream f(wav, std::ios::binary);
            const uint32_t n = (uint32_t)clip.samples.size(), data = n * 2, riff = 36 + data, rate = 16000, br = rate * 2;
            const uint16_t fmt = 1, ch = 1, ba = 2, bits = 16;
            const uint32_t sixteen = 16;
            f.write("RIFF", 4); f.write((const char*)&riff, 4); f.write("WAVEfmt ", 8); f.write((const char*)&sixteen, 4);
            f.write((const char*)&fmt, 2); f.write((const char*)&ch, 2); f.write((const char*)&rate, 4); f.write((const char*)&br, 4);
            f.write((const char*)&ba, 2); f.write((const char*)&bits, 2); f.write("data", 4); f.write((const char*)&data, 4);
            for (float x : clip.samples) {
                const int16_t s = (int16_t)std::lround(std::max(-1.0f, std::min(1.0f, x)) * 32767.0f);
                f.write((const char*)&s, 2);
            }
        }
        parakeet_ctx* actx = parakeet_capi_load(asr_path);
        parakeet_ctx* vctx = parakeet_capi_load(silero_path);
        CHECK(actx && vctx);
        if (actx && vctx) {
            char* j = parakeet_capi_transcribe_path_json_vad_with(actx, vctx, wav, 0, nullptr);
            CHECK(j != nullptr);
            if (j) {
                const std::string js(j);
                parakeet_capi_free_string(j);
                CHECK(js.find("\"words\"") != std::string::npos);
                // The words of the JSON text are those of the C++ call on the quantised clip.
                const size_t a = js.find("\"text\":\"");
                const size_t b = a == std::string::npos ? a : js.find('"', a + 8);
                const auto wj = words_of(a == std::string::npos ? "" : js.substr(a + 8, b - a - 8));
                const double d2 = (double)edit_distance(wj, ws) / (double)std::max<size_t>(1, ws.size());
                std::printf("C-API vs C++ word diff %.3f\n", d2);
                CHECK(!wj.empty() && d2 < 0.05);
            }
            // An option the segmenter takes: a 10 s cap gives shorter pieces and still text.
            char* j2 = parakeet_capi_transcribe_path_json_vad_with(actx, vctx, wav, 0, "{\"max_segment\":10}");
            CHECK(j2 != nullptr);
            if (j2) parakeet_capi_free_string(j2);
            CHECK(parakeet_capi_transcribe_path_json_vad_with(actx, vctx, wav, 0, "{\"bogus\":1}") == nullptr);
            CHECK(std::string(parakeet_capi_last_error(actx)).find("bogus") != std::string::npos);
            // vad_ctx must be a Silero context; an ASR context is refused.
            CHECK(parakeet_capi_transcribe_path_json_vad_with(actx, actx, wav, 0, nullptr) == nullptr);
            CHECK(parakeet_capi_transcribe_path_json_vad_with(actx, vctx, "/nonexistent.wav", 0, nullptr) == nullptr);
            // vad_ctx NULL: the model's own head; a model without one reports it.
            if (!m->config().vad.present) {
                CHECK(parakeet_capi_transcribe_path_json_vad_with(actx, nullptr, wav, 0, nullptr) == nullptr);
                CHECK(std::string(parakeet_capi_last_error(actx)) == "model has no VAD head");
            }
        }
        parakeet_capi_free(actx);
        parakeet_capi_free(vctx);
        std::remove(wav);
    }

    if (failures) return 1;
    std::puts("test_transcribe_vad_silero: OK");
    return 0;
}
