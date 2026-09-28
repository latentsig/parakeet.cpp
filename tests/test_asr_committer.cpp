// Unit test for pk::AsrCommitter with a fake transcriber (no model).
#include "asr_committer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace pk;
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

// The PCM value at each sample is its absolute time in seconds, so the fake
// knows where the buffer starts.
// Words start at `speech_from` seconds (silence before it).
static Transcriber fake(double speech_from = 0.0) {
    return [speech_from](const std::vector<float>& pcm) {
        std::vector<Word> w;
        if (pcm.empty()) return w;
        const double t0 = pcm[0];
        const double dur = pcm.size() / 16000.0;
        const int first = (int)std::ceil(std::max(t0, speech_from) / 0.5 - 1e-6);
        for (int i = first;; ++i) {
            const double s = i * 0.5 + 0.05 - t0, e = i * 0.5 + 0.45 - t0;
            if (e > dur) break;
            w.push_back({"w" + std::to_string(i), (float)s, (float)e, 0.9f});
        }
        return w;
    };
}
static std::vector<float> timeline(double a, double b) {
    std::vector<float> x;
    for (long i = std::lround(a * 16000); i < std::lround(b * 16000); ++i) x.push_back((float)(i / 16000.0));
    return x;
}

static void test_min_window() {
    AsrCommitter c(fake());
    auto x = timeline(0, 3);
    c.push(x.data(), (int)x.size());
    CHECK(c.commit(3.0, false).empty());       // 3 s < 4 s window
    CHECK(c.commit_sec() == 0.0);
}

static void test_right_context_and_resume() {
    AsrCommitter c(fake());
    auto x = timeline(0, 5);
    c.push(x.data(), (int)x.size());
    auto w = c.commit(5.0, false);
    // words must end <= 5 - 1 = 4 s: w0..w7 (w7 ends 3.95)
    CHECK(w.size() == 8);
    if (w.size() == 8) { CHECK(w.front().text == "w0"); CHECK(w.back().text == "w7"); }
    CHECK(std::fabs(c.commit_sec() - 3.95) < 1e-3);   // resumes after the last word
    auto y = timeline(5, 10);
    c.push(y.data(), (int)y.size());
    auto w2 = c.commit(10.0, false);
    CHECK(!w2.empty() && w2.front().text == "w8");     // no repeat, no gap
    CHECK(std::fabs(w2.front().start - 4.05f) < 1e-3); // absolute time
}

static void test_is_last_commits_all() {
    AsrCommitter c(fake());
    auto x = timeline(0, 2);
    c.push(x.data(), (int)x.size());
    auto w = c.commit(2.0, true);
    CHECK(w.size() == 4);                              // w0..w3, no right-context hold
}

static void test_duplicate_word_dropped() {
    // A transcriber that hears the last committed word again at the new start.
    int calls = 0;
    AsrCommitter c([&](const std::vector<float>& pcm) {
        ++calls;
        std::vector<Word> w;
        if (calls == 1) { w = {{"hello", 0.5f, 1.0f, 0.9f}, {"there", 1.1f, 1.5f, 0.9f}}; }
        else { w = {{"there", 0.0f, 0.2f, 0.9f}, {"friend", 0.4f, 0.9f, 0.9f}}; }
        (void)pcm;
        return w;
    });
    std::vector<float> x(16000 * 5, 0.0f);
    c.push(x.data(), (int)x.size());
    auto a = c.commit(5.0, false);
    CHECK(a.size() == 2);
    c.push(x.data(), (int)x.size());
    auto b = c.commit(10.0, true);
    CHECK(b.size() == 1 && b[0].text == "friend");
}

// A word whose reported span straddles the cut (its end lands past the right
// context, so it does not commit) must not have the cut land exactly on its
// start: the onset margin must back off from it, and the word must come back
// intact, at its correct absolute times, once more audio gives it context.
static void test_onset_margin_backoff() {
    AsrCommitter c([](const std::vector<float>& pcm) {
        std::vector<Word> w;
        if (pcm.empty()) return w;
        const double t0 = pcm[0];
        const double dur = pcm.size() / 16000.0;
        const double abs_start = 3.8, abs_end = 4.4;
        if (abs_start >= t0 && abs_end <= t0 + dur)
            w.push_back({"onset", (float)(abs_start - t0), (float)(abs_end - t0), 0.9f});
        return w;
    });
    auto x = timeline(0, 5);
    c.push(x.data(), (int)x.size());
    auto w = c.commit(5.0, false);
    CHECK(w.empty());                                  // straddles limit (4.0), so keep == 0
    CHECK(std::fabs(c.commit_sec() - 3.5) < 1e-3);      // 3.8 - kOnsetMargin(0.3), not 3.8
    // More audio gives the word its right context; it must return intact.
    auto y = timeline(5, 10);
    c.push(y.data(), (int)y.size());
    auto w2 = c.commit(10.0, true);
    CHECK(w2.size() == 1);
    if (w2.size() == 1) {
        CHECK(w2[0].text == "onset");
        CHECK(std::fabs(w2[0].start - 3.8f) < 1e-3);
        CHECK(std::fabs(w2[0].end - 4.4f) < 1e-3);
    }
}

// Non-speech: no words at all. The commit point must still advance and the
// buffer must not grow with the stream.
static void test_silence_released() {
    AsrCommitter c([](const std::vector<float>&) { return std::vector<Word>{}; });
    size_t max_buffered = 0;
    for (int s = 0; s < 20; ++s) {
        auto x = timeline(s, s + 1);
        c.push(x.data(), (int)x.size());
        CHECK(c.commit(s + 1.0, false).empty());
        max_buffered = std::max(max_buffered, c.buffered_samples());
    }
    CHECK(c.commit_sec() >= 15.0);                     // only the right context is held back
    CHECK(max_buffered <= (size_t)(5 * 16000));        // min window + one piece
    CHECK(c.buffered_samples() <= (size_t)(5 * 16000));
}

// Speech after a silent stretch commits every word once, at absolute times.
static void test_speech_after_silence() {
    AsrCommitter c(fake(10.0));
    std::vector<Word> all;
    for (int s = 0; s < 20; ++s) {
        auto x = timeline(s, s + 1);
        c.push(x.data(), (int)x.size());
        auto w = c.commit(s + 1.0, s == 19);
        all.insert(all.end(), w.begin(), w.end());
        CHECK(c.buffered_samples() <= (size_t)(5 * 16000));
    }
    CHECK(all.size() == 20);                           // w20..w39
    for (size_t i = 0; i < all.size(); ++i) {
        const int k = 20 + (int)i;
        CHECK(all[i].text == "w" + std::to_string(k));
        CHECK(std::fabs(all[i].start - (k * 0.5 + 0.05)) < 1e-3);
        CHECK(std::fabs(all[i].end - (k * 0.5 + 0.45)) < 1e-3);
    }
}

int main() {
    test_silence_released();
    test_speech_after_silence();
    test_min_window();
    test_right_context_and_resume();
    test_is_last_commits_all();
    test_duplicate_word_dropped();
    test_onset_margin_backoff();
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
