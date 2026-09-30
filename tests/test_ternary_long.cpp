// Packed ternary Redux through the local (banded) and chunked attention paths.
//
// Regression for a segfault on long audio: three attention `linear` lambdas
// (build_graph_batched_local, build_graph_local, build_graph_local_chunked)
// read only `<base>.weight`, which does not exist in a packed GGUF, and passed a
// null tensor to ggml_mul_mat. Long audio (encoder frames > 8192, about 655 s)
// switches to local attention automatically; PARAKEET_ATT_CONTEXT forces the same
// path on short audio, which is what this test does, so it stays fast.
//
// Env: PARAKEET_TEST_GGUF_REDUX_KEEP (--ternary keep), PARAKEET_TEST_GGUF_REDUX_DEQ.
// Optional: PARAKEET_TEST_LONG=1 also runs a ~700 s concatenation through the
// automatic trigger (slow, minutes).
#include "model.hpp"
#include "audio_io.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

using namespace pk;

static std::vector<std::string> words(const std::string& s) {
    std::istringstream is(s);
    std::vector<std::string> w;
    for (std::string t; is >> t;) w.push_back(t);
    return w;
}

static size_t word_edit_distance(const std::string& a, const std::string& b) {
    const auto x = words(a), y = words(b);
    std::vector<size_t> prev(y.size() + 1), cur(y.size() + 1);
    for (size_t j = 0; j <= y.size(); ++j) prev[j] = j;
    for (size_t i = 1; i <= x.size(); ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= y.size(); ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x[i - 1] != y[j - 1])});
        std::swap(prev, cur);
    }
    return prev[y.size()];
}

// Within 1 percent of the words, but never less than 2 words: a single split or
// join ("anymore" vs "any more") costs 2 word edits on a short clip.
static bool close_enough(const char* what, const std::string& deq, const std::string& keep) {
    const size_t d = word_edit_distance(deq, keep);
    const size_t n = words(deq).size();
    const size_t tol = std::max<size_t>(2, (n + 99) / 100);
    std::printf("  %s: word edit distance %zu of %zu words (tolerance %zu)\n", what, d, n, tol);
    if (d > tol) {
        std::fprintf(stderr, "FAIL: %s differs beyond tolerance\n  deq : %s\n  keep: %s\n",
                     what, deq.c_str(), keep.c_str());
        return false;
    }
    return true;
}

struct Result { std::string a, b, batch0, batch1; };

static bool run(const char* path, const std::vector<float>& sa, const std::vector<float>& sb,
                Result& r) {
    auto m = Model::load(path);
    if (!m) { std::fprintf(stderr, "FAIL: load %s\n", path); return false; }
    r.a = m->transcribe_pcm(sa, 16000, Decoder::kTDT);
    r.b = m->transcribe_pcm(sb, 16000, Decoder::kTDT);
    const auto batch = m->transcribe_pcm_batch({sa, sb}, 16000, Decoder::kTDT);
    if (batch.size() != 2) { std::fprintf(stderr, "FAIL: batch size\n"); return false; }
    r.batch0 = batch[0];
    r.batch1 = batch[1];
    return true;
}

int main() {
    const char* keep = std::getenv("PARAKEET_TEST_GGUF_REDUX_KEEP");
    const char* deq = std::getenv("PARAKEET_TEST_GGUF_REDUX_DEQ");
    if (!keep || !deq) {
        std::puts("skip: PARAKEET_TEST_GGUF_REDUX_KEEP / _DEQ unset");
        return 77;
    }
    Audio a, b;
    if (!load_audio_16k_mono("tests/fixtures/speech.wav", a) ||
        !load_audio_16k_mono("tests/fixtures/two_speakers.wav", b)) {
        std::fprintf(stderr, "FAIL: fixture load\n");
        return 1;
    }

    int failures = 0;
    // Force local attention (window 64) on short audio: single item goes through
    // build_graph_local_chunked, the batch through the batched local path.
    setenv("PARAKEET_ATT_CONTEXT", "64", 1);
    Result rd, rk;
    if (!run(deq, a.samples, b.samples, rd) || !run(keep, a.samples, b.samples, rk)) return 1;
    std::printf("local attention, packed: %s\n", rk.a.c_str());
    if (rk.a.empty() || rk.b.empty()) { std::fprintf(stderr, "FAIL: empty transcript\n"); ++failures; }
    if (!close_enough("single speech", rd.a, rk.a)) ++failures;
    if (!close_enough("single two_speakers", rd.b, rk.b)) ++failures;
    if (!close_enough("batched speech", rd.batch0, rk.batch0)) ++failures;
    if (!close_enough("batched two_speakers", rd.batch1, rk.batch1)) ++failures;
    unsetenv("PARAKEET_ATT_CONTEXT");

    if (const char* lg = std::getenv("PARAKEET_TEST_LONG"); lg && *lg == '1') {
        // Natural trigger: more than 8192 encoder frames (about 655 s).
        std::vector<float> longpcm;
        while (longpcm.size() < 16000u * 700u) {
            longpcm.insert(longpcm.end(), a.samples.begin(), a.samples.end());
            longpcm.insert(longpcm.end(), b.samples.begin(), b.samples.end());
        }
        std::string td, tk;
        { auto m = Model::load(deq);  td = m->transcribe_pcm(longpcm, 16000, Decoder::kTDT); }
        { auto m = Model::load(keep); tk = m->transcribe_pcm(longpcm, 16000, Decoder::kTDT); }
        if (!close_enough("long 700 s", td, tk)) ++failures;
    }

    if (failures) return 1;
    std::puts("test_ternary_long: OK");
    return 0;
}
