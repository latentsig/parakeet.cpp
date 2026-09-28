// Unit test for pk::SoundStream with a fake scorer (no model).
#include "sound_stream.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)
static bool near(float a, float b) { return std::fabs(a - b) < 1e-3f; }

// Class 0 score = fraction of samples == 1.0 (like CED's mean pooling).
static SoundScorer fake() {
    return [](const float* pcm, int n, std::vector<float>& p) {
        int on = 0;
        for (int i = 0; i < n; ++i) on += pcm[i] == 1.0f;
        p = {n > 0 ? (float)on / n : 0.0f, 0.0f};
        return true;
    };
}
// BRIEF DEFECT (see task-3-report.md): default SoundOpts::top_k is 5, but
// validate_sound_opts requires top_k <= n_classes, and every fake-scorer
// test below uses n_classes = 2 (fixed by fake()'s hardcoded 2-score
// output). SoundOpts{} therefore fails validation and SoundStream's
// constructor throws std::invalid_argument, uncaught, before any test body
// runs. top_k only feeds drain_windows()'s top-k queue, never the
// open/close hysteresis logic, so capping it to n_classes here changes no
// segment expectation in this file; it only makes construction valid.
static SoundOpts opts2() { SoundOpts o; o.top_k = 2; return o; }
// `sec` seconds of audio with the "sound" (1.0) in [a, b) seconds.
static std::vector<float> clip(float sec, float a, float b) {
    std::vector<float> x((size_t)(sec * 16000), 0.0f);
    for (size_t i = (size_t)(a * 16000); i < (size_t)(b * 16000) && i < x.size(); ++i) x[i] = 1.0f;
    return x;
}
static std::vector<SoundSegment> run_all(SoundStream& s, const std::vector<float>& x, int piece) {
    std::vector<SoundSegment> all;
    for (size_t i = 0; i < x.size(); i += piece) {
        const int n = (int)std::min<size_t>(piece, x.size() - i);
        const bool last = i + piece >= x.size();
        auto got = s.feed(x.data() + i, n, last);
        all.insert(all.end(), got.begin(), got.end());
    }
    return all;
}

// Sound in [5, 8): windows (3 s, hop 1 s) ending at 7 score 2/3 -> open at 6.
// Window ending 10 is [7, 10) = 1/3, still >= off 0.3. Window ending 11 is
// [8, 11) = 0 -> close at 8 + 1 = 9. Expected segment [6, 9), peak 1.0.
static void test_single_sound() {
    SoundStream s(fake(), 2, opts2());
    auto segs = run_all(s, clip(14.0f, 5.0f, 8.0f), 1600);  // 0.1 s pieces
    CHECK(segs.size() == 1);
    if (segs.size() == 1) {
        CHECK(segs[0].cls == 0);
        CHECK(near(segs[0].start, 6.0f));
        CHECK(near(segs[0].end, 9.0f));
        CHECK(near(segs[0].peak, 1.0f));
    }
}

// Piece size must not change the result.
static void test_piece_size_invariant() {
    const auto x = clip(14.0f, 5.0f, 8.0f);
    SoundStream a(fake(), 2, opts2()), b(fake(), 2, opts2());
    auto sa = run_all(a, x, 1600), sb = run_all(b, x, 16000 * 14);
    CHECK(sa.size() == sb.size());
    for (size_t i = 0; i < sa.size() && i < sb.size(); ++i)
        CHECK(near(sa[i].start, sb[i].start) && near(sa[i].end, sb[i].end));
}

// Hysteresis: a score that dips to 0.33 (above off 0.3) does not close.
static void test_hysteresis_holds() {
    SoundStream s(fake(), 2, opts2());
    auto x = clip(20.0f, 5.0f, 8.0f);
    for (size_t i = 9 * 16000; i < 10 * 16000; ++i) x[i] = 1.0f;  // second burst [9, 10)
    auto segs = run_all(s, x, 1600);
    CHECK(segs.size() == 1);  // one segment, not two
}

// Sound still present at the end: is_last closes it at the stream end.
static void test_is_last_closes_open() {
    SoundStream s(fake(), 2, opts2());
    auto segs = run_all(s, clip(8.0f, 5.0f, 8.0f), 1600);
    CHECK(segs.size() == 1);
    if (!segs.empty()) { CHECK(near(segs[0].start, 6.0f)); CHECK(near(segs[0].end, 8.0f)); }
    CHECK(s.finished());
}

// Short sounds are dropped by min_duration.
static void test_min_duration() {
    SoundOpts o = opts2(); o.min_duration_sec = 5.0f;
    SoundStream s(fake(), 2, o);
    CHECK(run_all(s, clip(14.0f, 5.0f, 8.0f), 1600).empty());
}

// Growing first windows: sound from 0 opens in the first window at 0.
static void test_growing_first_window() {
    SoundStream s(fake(), 2, opts2());
    auto segs = run_all(s, clip(10.0f, 0.0f, 2.0f), 1600);
    CHECK(segs.size() == 1);
    if (!segs.empty()) CHECK(near(segs[0].start, 0.0f));
}

// Very short input and empty feeds: no segments, no throw.
static void test_short_and_empty() {
    SoundStream s(fake(), 2, opts2());
    CHECK(s.feed(nullptr, 0, false).empty());
    auto x = clip(0.1f, 0.0f, 0.1f);
    CHECK(s.feed(x.data(), (int)x.size(), true).empty());
    CHECK(s.finished());
}

// Score queue: one window per hop, top-k sorted, drained once.
static void test_drain_windows() {
    SoundOpts o; o.top_k = 1;
    SoundStream s(fake(), 2, o);
    auto x = clip(4.0f, 0.0f, 4.0f);
    s.feed(x.data(), (int)x.size(), false);
    auto w = s.drain_windows();
    CHECK(w.size() == 4);  // windows ending at 1, 2, 3, 4 s
    if (w.size() == 4) {
        CHECK(near(w[0].start, 0.0f) && near(w[0].end, 1.0f));
        CHECK(near(w[3].start, 1.0f) && near(w[3].end, 4.0f));
        CHECK(w[3].top.size() == 1 && w[3].top[0].first == 0 && near(w[3].top[0].second, 1.0f));
    }
    CHECK(s.drain_windows().empty());
}

// open_segments reports a running class with end = time().
static void test_open_segments() {
    SoundStream s(fake(), 2, opts2());
    auto x = clip(7.0f, 5.0f, 7.0f);
    s.feed(x.data(), (int)x.size(), false);
    auto open = s.open_segments();
    CHECK(open.size() == 1);
    if (!open.empty()) CHECK(near(open[0].end, 7.0f));
    CHECK(s.safe_until() <= 6.0 + 1e-6);  // the open segment started at 6
}

static void test_opts_validation() {
    SoundOpts o;
    CHECK(validate_sound_opts(o, 527).empty());
    o = SoundOpts{}; o.hop_sec = 4.0f;           CHECK(!validate_sound_opts(o, 527).empty());
    o = SoundOpts{}; o.off_threshold = 0.5f;     CHECK(!validate_sound_opts(o, 527).empty());
    o = SoundOpts{}; o.hop_sec = 0.1f;           CHECK(!validate_sound_opts(o, 527).empty());
    o = SoundOpts{}; o.window_sec = 12.0f;       CHECK(!validate_sound_opts(o, 527).empty());
    o = SoundOpts{}; o.top_k = 600;              CHECK(!validate_sound_opts(o, 527).empty());
    bool threw = false;
    try { SoundStream bad(fake(), 2, o); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

// A failing scorer surfaces as std::runtime_error.
static void test_scorer_failure() {
    SoundStream s([](const float*, int, std::vector<float>&) { return false; }, 2, opts2());
    auto x = clip(2.0f, 0.0f, 0.0f);
    bool threw = false;
    try { s.feed(x.data(), (int)x.size(), false); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

// RULING (task-3-brief plan defect fix): the buffer trim must key off
// min(next_end_, samples_in_), not next_end_ alone, or an is_last tail
// window that ends between two hops reads before the buffer start.
// clip(8.5, 5.0, 8.5) fed in 0.1 s (1600-sample) pieces, is_last on the
// last piece: the last full hop window ends at 8 s (unscored tail 0.5 s
// < the next hop at 9 s), so at is_last the tail window is
// [8 - 3, 8.5) intersected with what has streamed = [5.5, 8.5) (0.16 s+
// tail, scored). The sound (still on at 8.5) is present throughout, so it
// stays open through 6 s (opened by the window ending at 7 s, 2/3 on) and
// is closed by is_last at the stream end, 8.5 s.
static void test_is_last_tail_window_after_hop_gap() {
    SoundStream s(fake(), 2, opts2());
    auto segs = run_all(s, clip(8.5f, 5.0f, 8.5f), 1600);  // 0.1 s pieces
    CHECK(segs.size() == 1);
    if (segs.size() == 1) {
        CHECK(segs[0].cls == 0);
        CHECK(near(segs[0].start, 6.0f));
        CHECK(near(segs[0].end, 8.5f));
    }
    CHECK(s.finished());
}

// safe_until() must be a true lower bound on every future segment's start,
// including across the is_last tail window (fix round 1: while streaming,
// safe_until() has to account for a tail window that could still open a
// class at max(0, samples_in_ - hop_n_), not just the last window actually
// scored). With the default 3 s window / 1 s hop, sound in [7.5, 8.5) only
// scores 0.5/3 = 0.167 in the regular window ending at 8 s, below
// on_threshold, so use a lower on_threshold (0.3) to open a class in the
// is_last tail window [5.5, 8.5) (score 1/3 = 0.333). Right before the
// final 0.1 s piece, samples_in_ = 8.4 s and scored_end_ = 8.0 s: the old
// safe_until() (= scored_end_ alone) said 8.0 s, but that final feed()
// call returns a segment starting at 7.5 s (the tail window's newest hop,
// max(0, samples_in_ - hop_n_) at samples_in_ = 8.5 s) -- a segment
// starting 0.5 s before what the old code promised. The fixed safe_until()
// says min(8.0, 8.4 - 1.0) = 7.4 s at that point, which 7.5 s honors.
static void test_safe_until_bounds_tail_open() {
    SoundOpts o = opts2(); o.on_threshold = 0.3f; o.off_threshold = 0.2f;
    SoundStream s(fake(), 2, o);
    auto x = clip(8.5f, 7.5f, 8.5f);
    const int piece = 1600;  // 0.1 s
    bool saw_segment = false;
    for (size_t i = 0; i < x.size(); i += piece) {
        const int n = (int)std::min<size_t>(piece, x.size() - i);
        const bool last = i + piece >= x.size();
        const double safe = s.safe_until();
        auto got = s.feed(x.data() + i, n, last);
        for (const auto& seg : got) {
            saw_segment = true;
            CHECK(seg.start >= safe - 1e-6);
        }
    }
    CHECK(saw_segment);  // otherwise the invariant above is checked vacuously
    CHECK(s.finished());
}

int main() {
    test_single_sound();
    test_piece_size_invariant();
    test_hysteresis_holds();
    test_is_last_closes_open();
    test_min_duration();
    test_growing_first_window();
    test_short_and_empty();
    test_drain_windows();
    test_open_segments();
    test_opts_validation();
    test_scorer_failure();
    test_is_last_tail_window_after_hop_gap();
    test_safe_until_bounds_tail_open();
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
