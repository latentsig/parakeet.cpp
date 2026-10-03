// Silero VAD model test. Skips (exit 77) without a model.
//
//   PARAKEET_TEST_SILERO_GGUF   silero-vad GGUF, F32 or F16 (see scripts/convert_silero_vad_to_gguf.py)
//
// Checks, for 16 kHz and 8 kHz:
//   - probabilities against tests/fixtures/silero_vad_ref.txt (onnxruntime, see
//     scripts/gen_silero_vad_ref.py); the clip is built here from speech.wav:
//     1.0 s silence + speech + 1.5 s silence; the 8 kHz clip keeps every second sample.
//     F32 files must agree within 1e-4, F16 files within 5e-3.
//   - streaming in arbitrary pieces gives the same bits as one call.
//   - reset(), flush(), the error paths and independent streams.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "audio_io.hpp"
#include "ggml.h"
#include "gguf.h"
#include "ggml_graph.hpp"
#include "silero_vad.hpp"

#ifndef PK_SOURCE_DIR
#define PK_SOURCE_DIR "."
#endif

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_SILERO_GGUF");
    if (!gguf) return 77;
    pk::set_num_threads(1);
    std::string err;
    auto vad = pk::SileroVad::load(gguf, &err);
    if (!vad) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    const bool f16 = std::string(gguf).find("f16") != std::string::npos;
    const double tol = f16 ? 5e-3 : 1e-4;
    CHECK(vad->supports(16000) && vad->supports(8000) && !vad->supports(44100));
    CHECK(vad->chunk_samples(16000) == 512 && vad->chunk_samples(8000) == 256 && vad->chunk_samples(22050) == 0);
    CHECK(vad->chunk_sec(16000) == pk::kSileroFrameSec && vad->chunk_sec(8000) == pk::kSileroFrameSec);
    CHECK(!vad->new_stream(44100).valid());
    CHECK(vad->probabilities(nullptr, 0, 44100).empty());

    // The clip.
    pk::Audio a;
    const std::string root = PK_SOURCE_DIR;
    CHECK(pk::load_audio_16k_mono(root + "/tests/fixtures/speech.wav", a));
    CHECK(a.samples.size() == 118960);
    std::vector<float> clip16(16000, 0.0f);
    clip16.insert(clip16.end(), a.samples.begin(), a.samples.end());
    clip16.resize(clip16.size() + 24000, 0.0f);
    std::vector<float> clip8;
    for (size_t i = 0; i < clip16.size(); i += 2) clip8.push_back(clip16[i]);

    // The reference.
    std::map<int, std::vector<float>> ref;
    {
        std::ifstream f(root + "/tests/fixtures/silero_vad_ref.txt");
        CHECK((bool)f);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream is(line);
            int sr = 0;
            double v = 0;
            if (is >> sr >> v) ref[sr].push_back((float)v);
        }
    }

    std::mt19937 rng(1);
    for (int sr : {16000, 8000}) {
        const std::vector<float>& clip = sr == 16000 ? clip16 : clip8;
        const size_t chunk = (size_t)vad->chunk_samples(sr);
        const std::vector<float> all = vad->probabilities(clip.data(), clip.size(), sr);
        CHECK(all.size() == (clip.size() + chunk - 1) / chunk);
        CHECK(clip.size() % chunk != 0);  // the last chunk is padded

        // Parity with onnxruntime.
        const std::vector<float>& r = ref[sr];
        CHECK(r.size() == all.size());
        double mx = 0.0;
        for (size_t i = 0; i < all.size() && i < r.size(); ++i) mx = std::fmax(mx, std::fabs((double)all[i] - r[i]));
        std::printf("%d Hz: max abs diff vs onnxruntime %.3g over %zu chunks (limit %.0e)\n", sr, mx, all.size(), tol);
        CHECK(mx < tol);

        // Silence at both ends, speech in the middle.
        CHECK(all.front() < 0.5f && all.back() < 0.5f);
        size_t speech = 0;
        for (float p : all) {
            CHECK(p >= 0.0f && p <= 1.0f);
            speech += p >= 0.5f;
        }
        CHECK(speech * 3 > all.size());

        // Streaming in arbitrary pieces equals the one-call result bit for bit.
        for (int trial = 0; trial < 6; ++trial) {
            pk::SileroVad::Stream s = vad->new_stream(sr);
            CHECK(s.valid() && s.sample_rate() == sr);
            std::vector<float> got;
            size_t pos = 0;
            while (pos < clip.size()) {
                size_t n = trial == 0 ? 1 : trial == 1 ? chunk : trial == 2 ? 7 * chunk / 2 : rng() % (3 * chunk);
                n = std::min(n, clip.size() - pos);
                CHECK(s.process_chunk(clip.data() + pos, n, &got));
                pos += n;
            }
            CHECK(s.pending_samples() == clip.size() % chunk);
            CHECK(s.flush(&got));
            CHECK(s.pending_samples() == 0);
            CHECK(same_bits(got, all));
        }

        // reset() restores the initial state; flush() with nothing pending adds nothing.
        pk::SileroVad::Stream s = vad->new_stream(sr);
        std::vector<float> first, again;
        CHECK(s.process_chunk(clip.data(), chunk * 40, &first) && first.size() == 40);
        const size_t before = first.size();
        CHECK(s.flush(&first) && first.size() == before);
        s.reset();
        CHECK(s.process_chunk(clip.data(), chunk * 40, &again));
        CHECK(same_bits(first, again));

        // Two streams on one model do not disturb each other.
        pk::SileroVad::Stream s1 = vad->new_stream(sr), s2 = vad->new_stream(sr);
        std::vector<float> p1, p2;
        for (size_t pos = 0; pos + chunk <= clip.size(); pos += chunk) {
            CHECK(s1.process_chunk(clip.data() + pos, chunk, &p1));
            CHECK(s2.process_chunk(clip.data() + (clip.size() - chunk - pos % (clip.size() - chunk)), chunk, &p2));
        }
        std::vector<float> only1;
        pk::SileroVad::Stream s3 = vad->new_stream(sr);
        CHECK(s3.process_chunk(clip.data(), clip.size() / chunk * chunk, &only1));
        CHECK(same_bits(p1, only1));
    }

    // Digital silence is not speech.
    std::vector<float> zeros(16000, 0.0f);
    for (float p : vad->probabilities(zeros.data(), zeros.size(), 16000)) CHECK(p < 0.5f);

    // A file with a changed hyper-parameter is refused with a message.
    {
        const std::string bad = "test_silero_vad.tmp.gguf";
        gguf_context* src = gguf_init_from_file(gguf, gguf_init_params{/*no_alloc*/ true, nullptr});
        CHECK(src != nullptr);
        gguf_context* g = gguf_init_empty();
        if (src) gguf_set_kv(g, src);
        if (src) {
            gguf_set_val_u32(g, "silero_vad.8000.stft.hop", 100);
            CHECK(gguf_write_to_file(g, bad.c_str(), false));
            gguf_free(g);
            gguf_free(src);
            std::string e;
            CHECK(!pk::SileroVad::load(bad, &e));
            CHECK(e.find("stft.hop") != std::string::npos);
            std::remove(bad.c_str());
        }
    }

    pk::shutdown_backend();
    if (failures) return 1;
    std::printf("test_silero_vad OK\n");
    return 0;
}
