// Standalone VAD through the C-API: speech regions on a synthetic
// speech/silence pattern, options, probabilities, resampling, the error paths,
// and several threads on one context and on separate contexts.
//
// LABEL model; run from the project root (fixtures are relative).
// Env:
//   PARAKEET_TEST_GGUF_ULTRA  a model with a VAD head (skip 77 if unset)
//   PARAKEET_TEST_GGUF_REDUX_KEEP  optional second model with a VAD head
//   PARAKEET_TEST_GGUF        a model without a VAD head (optional)
#include "parakeet_capi.h"
#include "audio_io.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

struct Seg { double s, e; };

// Reads the numbers after `"start":` / `"end":` pairs in the "segments" array.
static std::vector<Seg> segments_of(const std::string& j) {
    std::vector<Seg> out;
    size_t pos = j.find("\"segments\":[");
    if (pos == std::string::npos) return out;
    const size_t end = j.find(']', pos);
    while (true) {
        pos = j.find("\"start\":", pos);
        if (pos == std::string::npos || pos > end) break;
        Seg g{};
        g.s = std::atof(j.c_str() + pos + 8);
        const size_t q = j.find("\"end\":", pos);
        g.e = std::atof(j.c_str() + q + 6);
        out.push_back(g);
        pos = q + 6;
    }
    return out;
}

static std::vector<double> probs_of(const std::string& j) {
    std::vector<double> out;
    size_t pos = j.find("\"probabilities\":[");
    if (pos == std::string::npos) return out;
    pos += 17;
    while (j[pos] != ']') {
        char* e = nullptr;
        out.push_back(std::strtod(j.c_str() + pos, &e));
        pos = (size_t)(e - j.c_str());
        if (j[pos] == ',') ++pos;
    }
    return out;
}

static std::string take(char* p) {
    if (!p) return std::string();
    std::string s(p);
    parakeet_capi_free_string(p);
    return s;
}

// Total length of [a, b] covered by regions.
static double covered(const std::vector<Seg>& r, double a, double b) {
    double t = 0;
    for (const auto& g : r) t += std::max(0.0, std::min(g.e, b) - std::max(g.s, a));
    return t;
}

int main() {
    const char* ultra = std::getenv("PARAKEET_TEST_GGUF_ULTRA");
    if (!ultra) { std::puts("skip: PARAKEET_TEST_GGUF_ULTRA unset"); return 77; }
    const char* redux = std::getenv("PARAKEET_TEST_GGUF_REDUX_KEEP");
    const char* plain = std::getenv("PARAKEET_TEST_GGUF");

    // Synthetic clip: 2.0 s silence, speech, 2.5 s silence, speech, 2.0 s silence.
    pk::Audio sp;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", sp)) { std::fprintf(stderr, "no fixture\n"); return 1; }
    const double sp_len = (double)sp.samples.size() / 16000.0;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 0.0005f);
    auto silence = [&](double sec) { std::vector<float> v((size_t)(sec * 16000)); for (auto& x : v) x = nd(rng); return v; };
    std::vector<float> clip;
    auto add = [&](const std::vector<float>& v) { clip.insert(clip.end(), v.begin(), v.end()); };
    add(silence(2.0)); add(sp.samples); add(silence(2.5)); add(sp.samples); add(silence(2.0));
    const double a0 = 2.0, a1 = 2.0 + sp_len, b0 = a1 + 2.5, b1 = b0 + sp_len;
    const double total = (double)clip.size() / 16000.0;

    parakeet_ctx* ctx = parakeet_capi_load(ultra);
    if (!ctx) { std::fprintf(stderr, "load failed\n"); return 1; }

    // 1. Defaults: speech regions on the original timeline.
    const std::string j = take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, nullptr));
    std::printf("%s\n", j.c_str());
    CHECK(!j.empty());
    CHECK(j.find("\"mode\":\"speech\"") != std::string::npos);
    CHECK(j.find("\"frame_sec\":0.080") != std::string::npos);
    CHECK(j.find("\"backend\":\"") != std::string::npos);
    CHECK(j.find("\"probabilities\"") == std::string::npos);
    const std::vector<Seg> r = segments_of(j);
    CHECK(!r.empty());
    for (size_t i = 0; i < r.size(); ++i) {
        CHECK(r[i].e > r[i].s);
        CHECK(r[i].s >= 0 && r[i].e <= total + 0.01);
        if (i) CHECK(r[i].s >= r[i - 1].e);
    }
    const double tol = 0.4;  // the head is a 80 ms frame model; edges are soft
    CHECK(!r.empty() && std::fabs(r.front().s - a0) < tol);
    CHECK(!r.empty() && std::fabs(r.back().e - b1) < tol);
    // Mostly speech inside the speech spans, nothing inside the gaps.
    CHECK(covered(r, a0, a1) > 0.8 * sp_len);
    CHECK(covered(r, b0, b1) > 0.8 * sp_len);
    CHECK(covered(r, 0.0, a0 - tol) < 1e-9);
    CHECK(covered(r, a1 + tol, b0 - tol) < 1e-9);
    CHECK(covered(r, b1 + tol, total) < 1e-9);

    // 2. Options: probabilities and a stricter threshold.
    const std::string jp = take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000,
                                                          "{\"probabilities\": true, \"threshold\": 0.5}"));
    const std::vector<double> p = probs_of(jp);
    CHECK(std::llabs((long long)p.size() - (long long)std::ceil(total / 0.08)) <= 2);
    CHECK(segments_of(jp).size() == r.size());
    {
        double lo = 0, hi = 0;
        for (size_t i = 0; i < p.size(); ++i) { const double t = (double)i * 0.08; if (t > 0.2 && t < a0 - 0.3) lo += p[i]; if (t > a0 + 0.5 && t < a1 - 0.5) hi += p[i]; }
        CHECK(hi / std::max(1.0, (a1 - a0 - 1.0) / 0.08) > 0.5);
        CHECK(lo / std::max(1.0, (a0 - 0.5) / 0.08) < 0.3);
    }
    const std::string jhigh = take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, "{\"threshold\":0.999}"));
    CHECK(covered(segments_of(jhigh), 0, total) <= covered(r, 0, total) + 1e-9);
    // A huge min_pause merges everything between the first and last speech.
    const std::string jm = take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, "{\"min_pause\":4}"));
    CHECK(segments_of(jm).size() == 1);

    // 3. "segments" mode: the transcriber's cuts. A clip over the cap is cut at pauses.
    {
        std::vector<float> longclip;
        for (int i = 0; i < 4; ++i) {
            longclip.insert(longclip.end(), clip.begin(), clip.end());
        }
        const double lt = (double)longclip.size() / 16000.0;
        const std::string js = take(parakeet_capi_vad_pcm_json(ctx, longclip.data(), (int)longclip.size(), 16000,
                                                              "{\"mode\":\"segments\",\"max_segment\":15}"));
        CHECK(js.find("\"mode\":\"segments\"") != std::string::npos);
        const auto sg = segments_of(js);
        CHECK(sg.size() >= 4);
        for (const auto& g : sg) CHECK(g.e - g.s <= 15.0 + 1e-6);
        CHECK(covered(sg, 0, lt) > 0.8 * 8 * sp_len);
        // Audio within the cap is returned whole, as the transcriber does.
        const std::string jw = take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, "{\"mode\":\"segments\"}"));
        const auto w = segments_of(jw);
        CHECK(w.size() == 1 && w[0].s == 0.0 && std::fabs(w[0].e - total) < 0.002);
    }

    // 4. Other sample rates: 8 kHz decimation keeps the timeline.
    {
        std::vector<float> c8;
        for (size_t i = 0; i + 1 < clip.size(); i += 2) c8.push_back(0.5f * (clip[i] + clip[i + 1]));
        const auto r8 = segments_of(take(parakeet_capi_vad_pcm_json(ctx, c8.data(), (int)c8.size(), 8000, nullptr)));
        CHECK(!r8.empty() && std::fabs(r8.front().s - a0) < 0.6 && std::fabs(r8.back().e - b1) < 0.6);
    }

    // 5. File variant equals PCM variant.
    {
        const std::string jf = take(parakeet_capi_vad_path_json(ctx, "tests/fixtures/speech.wav", nullptr));
        const std::string jq = take(parakeet_capi_vad_pcm_json(ctx, sp.samples.data(), (int)sp.samples.size(), 16000, nullptr));
        CHECK(!jf.empty() && jf == jq);
        CHECK(!segments_of(jf).empty());
    }

    // 6. Edge cases and errors.
    {
        const std::string je = take(parakeet_capi_vad_pcm_json(ctx, nullptr, 0, 16000, nullptr));
        CHECK(!je.empty() && segments_of(je).empty());
        std::vector<float> tiny(800, 0.0f);  // 50 ms
        CHECK(parakeet_capi_vad_pcm_json(ctx, tiny.data(), (int)tiny.size(), 16000, nullptr) != nullptr ||
              std::strlen(parakeet_capi_last_error(ctx)) > 0);
        const char* bad_opts[] = {"{\"threshold\":0}", "{\"threshold\":1.5}", "{\"min_pause\":-1}",
                                  "{\"mode\":\"x\"}", "{\"nope\":1}", "{\"threshold\":", "[1]",
                                  "{\"probabilities\":1}", "{} x"};
        for (const char* o : bad_opts) {
            CHECK(parakeet_capi_vad_pcm_json(ctx, sp.samples.data(), (int)sp.samples.size(), 16000, o) == nullptr);
            CHECK(std::strlen(parakeet_capi_last_error(ctx)) > 0);
        }
        CHECK(parakeet_capi_vad_pcm_json(ctx, nullptr, 5, 16000, nullptr) == nullptr);
        CHECK(parakeet_capi_vad_pcm_json(ctx, sp.samples.data(), 5, 0, nullptr) == nullptr);
        CHECK(parakeet_capi_vad_path_json(ctx, nullptr, nullptr) == nullptr);
        CHECK(parakeet_capi_vad_path_json(ctx, "/nonexistent.wav", nullptr) == nullptr);
        CHECK(parakeet_capi_vad_pcm_json(nullptr, sp.samples.data(), 5, 16000, nullptr) == nullptr);
        CHECK(!take(parakeet_capi_vad_pcm_json(ctx, sp.samples.data(), (int)sp.samples.size(), 16000, "  {}  ")).empty());
        CHECK(std::strlen(parakeet_capi_last_error(ctx)) == 0);
    }

    // 7. Threads: one context from 4 threads, and one context per thread. Same result as serial.
    {
        std::vector<std::string> out(4);
        std::vector<std::thread> th;
        for (int t = 0; t < 4; ++t)
            th.emplace_back([&, t] { out[t] = take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, nullptr)); });
        for (auto& x : th) x.join();
        for (int t = 0; t < 4; ++t) CHECK(out[t] == j);
        std::vector<std::string> out2(3);
        th.clear();
        for (int t = 0; t < 3; ++t)
            th.emplace_back([&, t] {
                parakeet_ctx* c = parakeet_capi_load(ultra);
                if (c) { out2[t] = take(parakeet_capi_vad_pcm_json(c, clip.data(), (int)clip.size(), 16000, nullptr)); parakeet_capi_free(c); }
            });
        for (auto& x : th) x.join();
        for (int t = 0; t < 3; ++t) CHECK(out2[t] == j);
        // With the backend pool on, concurrent calls still agree.
        parakeet_capi_set_concurrency(ctx, 2, 2);
        th.clear();
        for (int t = 0; t < 4; ++t)
            th.emplace_back([&, t] { out[t] = take(parakeet_capi_vad_pcm_json(ctx, clip.data(), (int)clip.size(), 16000, nullptr)); });
        for (auto& x : th) x.join();
        for (int t = 0; t < 4; ++t) CHECK(out[t] == j);
        parakeet_capi_set_concurrency(ctx, 1, 0);
    }
    parakeet_capi_free(ctx);

    // 8. Packed Redux (CPU only): same document shape, same backend report.
    if (redux) {
        parakeet_ctx* rc = parakeet_capi_load(redux);
        CHECK(rc != nullptr);
        if (rc) {
            const std::string jr = take(parakeet_capi_vad_pcm_json(rc, clip.data(), (int)clip.size(), 16000, nullptr));
            std::printf("redux: %s\n", jr.c_str());
            CHECK(jr.find("\"backend\":\"cpu\"") != std::string::npos);
            const auto rr = segments_of(jr);
            CHECK(!rr.empty() && std::fabs(rr.front().s - a0) < 0.5 && std::fabs(rr.back().e - b1) < 0.5);
            parakeet_capi_free(rc);
        }
    }

    // 9. A model without the head fails cleanly.
    if (plain) {
        parakeet_ctx* pc = parakeet_capi_load(plain);
        CHECK(pc != nullptr);
        if (pc) {
            CHECK(parakeet_capi_vad_pcm_json(pc, clip.data(), (int)clip.size(), 16000, nullptr) == nullptr);
            CHECK(std::string(parakeet_capi_last_error(pc)) == "model has no VAD head");
            CHECK(parakeet_capi_vad_path_json(pc, "tests/fixtures/speech.wav", nullptr) == nullptr);
            CHECK(std::string(parakeet_capi_last_error(pc)) == "model has no VAD head");
            parakeet_capi_free(pc);
        }
    }

    if (failures) return 1;
    std::puts("test_capi_vad: OK");
    return 0;
}
