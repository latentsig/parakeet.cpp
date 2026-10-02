// Stress test for the backend pool: N threads, K backends, a fixed clip set.
// Every result must equal the single-thread result with no pool: text, token
// ids, frames, spans, confidences and word timestamps, compared exactly.
//
// Models: PARAKEET_TEST_GGUF, PARAKEET_TEST_GGUF_ULTRA,
// PARAKEET_TEST_GGUF_REDUX_KEEP (each one that is set is tested).
// Skips (77) when none is set. PARAKEET_STRESS_QUICK=1 shortens the clips,
// rounds and configurations (for slow builds such as ThreadSanitizer). Run it under ThreadSanitizer too (see
// docs/concurrency.md).
//
// LABEL model
// WORKING_DIRECTORY (tests run from the project root)
#include "audio_io.hpp"
#include "model.hpp"
#include "parakeet_capi.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

static std::atomic<int> g_failures{0};

static bool same(const pk::Transcription& a, const pk::Transcription& b) {
    if (a.text != b.text) return false;
    if (a.tokens.size() != b.tokens.size() || a.words.size() != b.words.size()) return false;
    for (size_t i = 0; i < a.tokens.size(); ++i) {
        const auto &x = a.tokens[i], &y = b.tokens[i];
        if (x.id != y.id || x.frame != y.frame || x.span != y.span || x.conf != y.conf) return false;
    }
    for (size_t i = 0; i < a.words.size(); ++i) {
        const auto &x = a.words[i], &y = b.words[i];
        if (x.text != y.text || x.start != y.start || x.end != y.end || x.conf != y.conf) return false;
    }
    return true;
}

struct Clip { std::string name; std::vector<float> pcm; };

static bool quick() { return std::getenv("PARAKEET_STRESS_QUICK") != nullptr; }

static std::vector<Clip> load_clips() {
    std::vector<Clip> clips;
    if (quick()) {
        // Short clips of different lengths (the backends size their buffers
        // differently) keep a slow build fast.
        pk::Audio sp, two, clip;
        if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", sp) ||
            !pk::load_audio_16k_mono("tests/fixtures/two_speakers.wav", two) ||
            !pk::load_audio_16k_mono("tests/fixtures/clip.wav", clip)) {
            std::fprintf(stderr, "cannot load fixtures\n");
            std::exit(1);
        }
        clips.push_back({"speech_3s", std::vector<float>(sp.samples.begin(), sp.samples.begin() + 3 * 16000)});
        clips.push_back({"speech_tail", std::vector<float>(sp.samples.end() - 5 * 16000 / 2, sp.samples.end())});
        clips.push_back({"clip", clip.samples});
        clips.push_back({"two_4s", std::vector<float>(two.samples.begin(), two.samples.begin() + 4 * 16000)});
        return clips;
    }
    for (const char* n : {"speech", "two_speakers", "clip"}) {
        pk::Audio a;
        if (!pk::load_audio_16k_mono(std::string("tests/fixtures/") + n + ".wav", a) || a.samples.empty()) {
            std::fprintf(stderr, "cannot load fixture %s\n", n);
            std::exit(1);
        }
        clips.push_back({n, a.samples});
    }
    // Different lengths make the backends grow their graph buffers differently.
    const auto& s = clips[0].pcm;
    clips.push_back({"speech_3_4", std::vector<float>(s.begin(), s.begin() + s.size() * 3 / 4)});
    clips.push_back({"speech_tail", std::vector<float>(s.begin() + s.size() / 3, s.end())});
    const auto& t = clips[1].pcm;
    clips.push_back({"two_head", std::vector<float>(t.begin(), t.begin() + t.size() / 2)});
    return clips;
}

static bool run_model(const char* path) {
    std::fprintf(stderr, "model %s\n", path);
    auto model = pk::Model::load(path);
    if (!model) { std::fprintf(stderr, "load failed\n"); return false; }
    const std::vector<Clip> clips = load_clips();
    const size_t nc = clips.size();

    // Reference: single thread, no pool.
    std::vector<pk::Transcription> ref(nc);
    for (size_t i = 0; i < nc; ++i) ref[i] = model->transcribe_with_timestamps(clips[i].pcm, 16000);
    std::vector<std::string> ref_text(nc);
    for (size_t i = 0; i < nc; ++i) ref_text[i] = model->transcribe_pcm(clips[i].pcm, 16000);
    std::vector<std::vector<float>> batch_in;
    for (const Clip& c : clips) batch_in.push_back(c.pcm);
    const std::vector<std::string> ref_batch = model->transcribe_pcm_batch(batch_in, 16000);

    struct Cfg { int threads, backends, each; };
    const Cfg full[] = {{4, 2, 1}, {8, 4, 1}, {6, 3, 2}, {3, 1, 0}, {5, 8, 1}};
    const Cfg fast[] = {{4, 2, 1}, {3, 3, 1}};
    const Cfg* cfgs = quick() ? fast : full;
    const size_t ncfg = quick() ? 2 : 5;
    const int rounds = quick() ? 1 : 3;
    bool all_ok = true;
    for (size_t ci = 0; ci < ncfg; ++ci) {
        const Cfg& cfg = cfgs[ci];
        const int eff = model->set_concurrency(cfg.backends, cfg.each);
        std::fprintf(stderr, "  threads=%d backends=%d (effective %d) threads_each=%d\n",
                     cfg.threads, cfg.backends, eff, cfg.each);
        std::atomic<int> bad{0};
        std::vector<std::thread> ts;
        for (int t = 0; t < cfg.threads; ++t)
            ts.emplace_back([&, t] {
                try {
                    for (int round = 0; round < rounds; ++round)
                        for (size_t k = 0; k < nc; ++k) {
                            const size_t i = (k + (size_t)t + (size_t)round) % nc;
                            pk::Transcription r = model->transcribe_with_timestamps(clips[i].pcm, 16000);
                            if (!same(r, ref[i])) {
                                std::fprintf(stderr, "    MISMATCH timestamps clip %s (thread %d)\n",
                                             clips[i].name.c_str(), t);
                                ++bad;
                            }
                            if ((k + (size_t)t) % 3 == 0) {
                                if (model->transcribe_pcm(clips[i].pcm, 16000) != ref_text[i]) {
                                    std::fprintf(stderr, "    MISMATCH text clip %s\n", clips[i].name.c_str());
                                    ++bad;
                                }
                            }
                        }
                    // The batched path also leases a backend.
                    if (model->transcribe_pcm_batch(batch_in, 16000) != ref_batch) {
                        std::fprintf(stderr, "    MISMATCH batch (thread %d)\n", t);
                        ++bad;
                    }
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "    exception: %s\n", e.what());
                    ++bad;
                }
            });
        for (auto& th : ts) th.join();
        if (bad != 0) { std::fprintf(stderr, "  FAIL: %d mismatches\n", bad.load()); all_ok = false; }
        std::fprintf(stderr, "  pool working set: %.1f MB\n",
                     (double)model->pool_working_set_bytes() / 1e6);
    }

    // Replacing the pool while no request runs, then back to no pool, still
    // gives the reference result.
    model->set_concurrency(2, 1);
    model->set_concurrency(1);
    pk::Transcription again = model->transcribe_with_timestamps(clips[0].pcm, 16000);
    if (!same(again, ref[0])) { std::fprintf(stderr, "  FAIL: result differs after the pool is removed\n"); all_ok = false; }

    // Replace the pool while requests are running: calls in flight finish on the
    // old pool and every result stays equal to the reference.
    {
        std::atomic<bool> stop{false};
        std::atomic<int> bad{0};
        model->set_concurrency(2, 1);
        std::vector<std::thread> ts;
        for (int t = 0; t < 3; ++t)
            ts.emplace_back([&, t] {
                size_t k = (size_t)t;
                while (!stop.load()) {
                    const size_t i = k++ % nc;
                    if (!same(model->transcribe_with_timestamps(clips[i].pcm, 16000), ref[i])) ++bad;
                }
            });
        for (int i = 0; i < 6; ++i) model->set_concurrency(2 + (i % 3), 1);
        stop = true;
        for (auto& th : ts) th.join();
        if (bad != 0) { std::fprintf(stderr, "  FAIL: %d mismatches while replacing the pool\n", bad.load()); all_ok = false; }
    }
    return all_ok;
}

int main() {
    const char* envs[] = {"PARAKEET_TEST_GGUF", "PARAKEET_TEST_GGUF_ULTRA", "PARAKEET_TEST_GGUF_REDUX_KEEP"};
    bool ran = false, ok = true;
    for (const char* e : envs) {
        const char* p = std::getenv(e);
        if (!p) continue;
        ran = true;
        if (!run_model(p)) ok = false;
    }
    if (!ran) return 77;
    std::fprintf(stderr, ok ? "test_pool_stress: ok\n" : "test_pool_stress: FAILED\n");
    return ok ? 0 : 1;
}
