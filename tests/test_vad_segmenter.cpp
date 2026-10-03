#include "vad_segmenter.hpp"

#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <random>
#include <utility>
#include <vector>

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static const double FS = 0.08;

// n frames of speech (p = 0.95) with silent ranges [a, b) in frames (p = 0.02).
static std::vector<float> make_p(int n, std::initializer_list<std::pair<int, int>> silences) {
    std::vector<float> p((size_t)n, 0.95f);
    for (auto s : silences)
        for (int i = s.first; i < s.second && i < n; ++i) p[(size_t)i] = 0.02f;
    return p;
}

// Segments are ordered, disjoint, inside [0, total], at most max_seg long, and
// every internal boundary is a whole number of frames. (Segments without
// speech are dropped, so consecutive segments need not touch.)
static void check_ordered(const std::vector<VadSegment>& s, double total, double max_seg) {
    double prev_end = 0.0;
    for (const auto& g : s) {
        CHECK(g.start >= prev_end - 1e-9);
        CHECK(g.end > g.start);
        CHECK(g.end <= total + 1e-9);
        CHECK(g.end - g.start <= max_seg + 1e-9);
        prev_end = g.end;
    }
}

static bool near(double a, double b) { return std::fabs(a - b) < 1e-6; }

static void test_defaults() {
    SegmenterOpts o;
    CHECK(o.threshold == 0.5f);
    CHECK(near(o.max_seg_sec, 30.0));
    CHECK(near(o.min_pause_sec, 0.2));
    CHECK(near(o.min_seg_sec, 1.0));
    CHECK(near(o.bridge_sec, 0.1));
    CHECK(near(o.min_speech_sec, 0.1));
}

static void test_short_is_single() {
    SegmenterOpts o;
    auto s = segment_by_vad(make_p(250, {}), 20.0, o);
    CHECK(s.size() == 1 && s[0].start == 0.0 && s[0].end == 20.0);
    s = segment_by_vad({}, 0.5, o);            // shorter than one frame
    CHECK(s.size() == 1 && s[0].end == 0.5);
    s = segment_by_vad(make_p(375, {}), 30.0, o);  // exactly the cap
    CHECK(s.size() == 1 && s[0].end == 30.0);
    // Audio up to the cap goes through whole even when it has no speech.
    s = segment_by_vad(make_p(300, {{0, 300}}), 24.0, o);
    CHECK(s.size() == 1 && s[0].start == 0.0 && near(s[0].end, 24.0));
}

static void test_threshold_is_inclusive() {
    SegmenterOpts o;
    // 70 s of p == 0.5 exactly is all speech: hard cuts, nothing dropped.
    std::vector<float> p(875, 0.5f);
    auto s = segment_by_vad(p, 70.0, o);
    CHECK(s.size() == 3);
    // Just below the threshold is silence: every segment is dropped.
    std::vector<float> q(875, 0.4999f);
    CHECK(segment_by_vad(q, 70.0, o).empty());
}

static void test_all_speech_hard_cuts() {
    SegmenterOpts o;
    const double total = 70.0;
    auto s = segment_by_vad(make_p((int)std::ceil(total / FS), {}), total, o);
    check_ordered(s, total, o.max_seg_sec);
    CHECK(s.size() == 3);
    if (s.size() == 3) {
        CHECK(near(s[0].start, 0.0) && near(s[0].end, 30.0));
        CHECK(near(s[1].start, 30.0) && near(s[1].end, 60.0));
        CHECK(near(s[2].start, 60.0) && near(s[2].end, 70.0));
    }
}

static void test_cuts_land_in_pauses() {
    SegmenterOpts o;
    const double total = 70.0;
    // pauses at [312,325) and [640,653): midpoints 318 (25.44 s) and 646 (51.68 s)
    auto p = make_p((int)std::ceil(total / FS), {{312, 325}, {640, 653}});
    auto s = segment_by_vad(p, total, o);
    check_ordered(s, total, o.max_seg_sec);
    CHECK(s.size() == 3);
    if (s.size() == 3) {
        CHECK(near(s[0].end, 318 * FS));
        CHECK(near(s[1].start, 318 * FS));
        CHECK(near(s[1].end, 646 * FS));
        CHECK(near(s[2].end, total));
    }
}

static void test_picks_last_pause_not_longest() {
    SegmenterOpts o;
    const double total = 40.0;
    // A long pause at 10-12 s and a short one (0.4 s) at 27 s. The last one wins.
    auto p = make_p((int)std::ceil(total / FS), {{125, 150}, {337, 342}});
    auto s = segment_by_vad(p, total, o);
    check_ordered(s, total, o.max_seg_sec);
    CHECK(s.size() == 2);
    if (s.size() == 2) CHECK(near(s[0].end, 339 * FS));  // (337 + 342) / 2
}

static void test_pause_must_be_fully_inside() {
    SegmenterOpts o;
    const double total = 45.0;
    // Pause 1 [200,230) is fully inside the window [0, 375]. Pause 2 [370,390)
    // starts inside but ends after 30 s, so it is not fully inside: pause 1 wins
    // even though pause 2 is later.
    auto p = make_p((int)std::ceil(total / FS), {{200, 230}, {370, 390}});
    auto s = segment_by_vad(p, total, o);
    check_ordered(s, total, o.max_seg_sec);
    CHECK(s.size() == 2);
    if (s.size() == 2) CHECK(near(s[0].end, 215 * FS));
}

static void test_midpoint_fallback() {
    SegmenterOpts o;
    const double total = 45.0;
    // Only pause: [350,390) = 28.0 to 31.2 s. It is not fully inside the 30 s
    // window, but its midpoint (370 = 29.6 s) is, so the cut goes there.
    auto p = make_p((int)std::ceil(total / FS), {{350, 390}});
    auto s = segment_by_vad(p, total, o);
    check_ordered(s, total, o.max_seg_sec);
    CHECK(s.size() == 2);
    if (s.size() == 2) CHECK(near(s[0].end, 370 * FS));
    // Midpoint beyond 30 s (pause [370,430), midpoint 400): hard cut at 30 s.
    p = make_p((int)std::ceil(total / FS), {{370, 430}});
    s = segment_by_vad(p, total, o);
    CHECK(s.size() == 2);
    if (s.size() == 2) CHECK(near(s[0].end, 30.0));
}

static void test_min_segment_one_second() {
    SegmenterOpts o;
    const double total = 31.0;
    // Pause [6,20): starts before 1 s, midpoint 13 = 1.04 s is inside: cut there.
    auto p = make_p((int)std::ceil(total / FS), {{6, 20}});
    auto s = segment_by_vad(p, total, o);
    CHECK(s.size() == 2);
    if (s.size() == 2) CHECK(near(s[0].end, 13 * FS));
    // Pause [2,14): midpoint 8 = 0.64 s is before 1 s: hard cut at 30 s.
    p = make_p((int)std::ceil(total / FS), {{2, 14}});
    s = segment_by_vad(p, total, o);
    CHECK(s.size() == 2 && near(s[0].end, 30.0));
}

static void test_min_pause() {
    SegmenterOpts o;
    const double total = 45.0;
    // 2 frames (0.16 s) is shorter than 0.2 s: hard cut at 30 s.
    auto s = segment_by_vad(make_p((int)std::ceil(total / FS), {{312, 314}}), total, o);
    CHECK(s.size() == 2 && near(s[0].end, 30.0));
    // 3 frames (0.24 s) is long enough.
    s = segment_by_vad(make_p((int)std::ceil(total / FS), {{312, 315}}), total, o);
    CHECK(s.size() == 2 && near(s[0].end, 313 * FS));
}

static void test_bridge_short_speech_gap() {
    SegmenterOpts o;
    const double total = 45.0;
    // A one-frame (0.08 s) dip inside speech is bridged, so it cannot form a
    // pause even when min_pause is set below one frame.
    o.min_pause_sec = 0.01;
    auto p = make_p((int)std::ceil(total / FS), {{312, 313}});
    auto s = segment_by_vad(p, total, o);
    CHECK(s.size() == 2 && near(s[0].end, 30.0));
    // A two-frame (0.16 s) gap is not bridged and is a pause at that setting.
    p = make_p((int)std::ceil(total / FS), {{312, 314}});
    s = segment_by_vad(p, total, o);
    CHECK(s.size() == 2 && near(s[0].end, 313 * FS));
}

static void test_drop_short_speech_runs() {
    SegmenterOpts o;
    const double total = 40.0;
    const int n = (int)std::ceil(total / FS);
    // Speech 0-5 s, then a lone one-frame blip at 34 s, silence elsewhere. The
    // 34 s blip (0.08 s) is shorter than 0.1 s and counts as no speech.
    std::vector<float> p((size_t)n, 0.02f);
    for (int i = 0; i < 62; ++i) p[(size_t)i] = 0.95f;
    p[425 - 1] = 0.95f;
    auto s = segment_by_vad(p, total, o);
    CHECK(s.size() == 1);
    if (!s.empty()) CHECK(s[0].start == 0.0);
    // A blip, a one-frame gap and a blip are bridged into a 3-frame run (0.24 s)
    // and kept as speech.
    p[425 - 1] = 0.95f; p[425 + 1] = 0.95f;
    s = segment_by_vad(p, total, o);
    CHECK(s.size() == 2);
    // Two adjacent frames (0.16 s) are speech by themselves.
    p[425 + 1] = 0.02f; p[425] = 0.95f;
    s = segment_by_vad(p, total, o);
    CHECK(s.size() == 2);
}

static void test_segments_without_speech_are_dropped() {
    SegmenterOpts o;
    const double total = 70.0;
    const int n = (int)std::ceil(total / FS);
    // Speech for the first 10 s, then nothing. The pause [125,875) is not inside
    // the first window, its midpoint (500 = 40 s) is beyond 30 s: hard cut at
    // 30 s. [30,60) and the trailing remainder [60,70) have no speech.
    std::vector<float> p((size_t)n, 0.02f);
    for (int i = 0; i < 125; ++i) p[(size_t)i] = 0.95f;
    auto s = segment_by_vad(p, total, o);
    check_ordered(s, total, o.max_seg_sec);
    CHECK(s.size() == 1);
    if (s.size() == 1) CHECK(near(s[0].start, 0.0) && near(s[0].end, 30.0));
    // Speech at the start of a later segment and silence in front of it.
    std::vector<float> q((size_t)n, 0.02f);
    for (int i = 500; i < 560; ++i) q[(size_t)i] = 0.95f;  // 40 s to 44.8 s
    s = segment_by_vad(q, total, o);
    check_ordered(s, total, o.max_seg_sec);
    CHECK(s.size() == 1);
    if (s.size() == 1) CHECK(s[0].start > 0.0 && s[0].end > 44.0);
}

static void test_all_silence() {
    SegmenterOpts o;
    const double total = 65.0;
    CHECK(segment_by_vad(make_p((int)std::ceil(total / FS), {{0, 100000}}), total, o).empty());
    CHECK(segment_by_vad({}, 65.0, o).empty());
}

static void test_random_property() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uniform_int_distribution<int> run_len(1, 12);
    SegmenterOpts o;
    for (int trial = 0; trial < 200; ++trial) {
        const double total = 31.0 + 600.0 * u(rng);
        const int n = (int)std::ceil(total / FS);
        std::vector<float> p((size_t)n, 0.9f);
        for (int i = 0; i < n; ) {
            if (u(rng) < 0.2f) {  // start a silence run
                int len = run_len(rng);
                for (int j = 0; j < len && i + j < n; ++j) p[(size_t)(i + j)] = 0.05f;
                i += len;
            } else {
                ++i;
            }
        }
        const auto s = segment_by_vad(p, total, o);
        check_ordered(s, total, o.max_seg_sec);
        CHECK(!s.empty());
        for (const auto& g : s) {
            const double q = g.start / FS;
            CHECK(std::fabs(q - std::round(q)) < 1e-6);  // whole frames
        }
    }
}

static void test_degenerate_opts() {
    const double bad[] = {0.0, -0.08, INFINITY, NAN};
    for (double fs : bad) {
        SegmenterOpts o;
        o.frame_sec = fs;
        auto s = segment_by_vad(std::vector<float>(100, 0.0f), 100.0, o);
        CHECK(s.size() == 1);
        if (s.size() == 1) CHECK(s[0].start == 0.0 && s[0].end == 100.0);
    }
    SegmenterOpts o;
    o.max_seg_sec = 0.05;  // smaller than a frame
    auto s = segment_by_vad(std::vector<float>(100, 0.0f), 100.0, o);
    CHECK(s.size() == 1);
    o = SegmenterOpts();
    o.max_seg_sec = NAN;
    s = segment_by_vad(std::vector<float>(100, 0.0f), 100.0, o);
    CHECK(s.size() == 1);
}

static void test_nonfinite_and_huge_opts() {
    const double bads[] = {INFINITY, -INFINITY, NAN, 1e300, 1e7};
    for (double v : bads) {
        for (int which = 0; which < 5; ++which) {
            SegmenterOpts o;
            if (which == 0) o.min_seg_sec = v;
            else if (which == 1) o.min_pause_sec = v;
            else if (which == 2) o.max_seg_sec = v;
            else if (which == 3) o.bridge_sec = v;
            else o.min_speech_sec = v;
            auto s = segment_by_vad(std::vector<float>(2000, 0.0f), 100.0, o);
            CHECK(s.size() == 1);
            if (s.size() == 1) CHECK(s[0].start == 0.0 && s[0].end == 100.0);
        }
    }
    const float thr[] = {INFINITY, -INFINITY, NAN};
    for (float t : thr) {
        SegmenterOpts o;
        o.threshold = t;
        auto s = segment_by_vad(std::vector<float>(2000, 0.0f), 100.0, o);
        CHECK(s.size() == 1);
        if (s.size() == 1) CHECK(s[0].start == 0.0 && s[0].end == 100.0);
    }
}

int main() {
    test_defaults();
    test_nonfinite_and_huge_opts();
    test_degenerate_opts();
    test_short_is_single();
    test_threshold_is_inclusive();
    test_all_speech_hard_cuts();
    test_cuts_land_in_pauses();
    test_picks_last_pause_not_longest();
    test_pause_must_be_fully_inside();
    test_midpoint_fallback();
    test_min_segment_one_second();
    test_min_pause();
    test_bridge_short_speech_gap();
    test_drop_short_speech_runs();
    test_segments_without_speech_are_dropped();
    test_all_silence();
    test_random_property();
    if (failures) return 1;
    std::puts("test_vad_segmenter: OK");
    return 0;
}
