#include "vad_segmenter.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <random>
#include <utility>
#include <vector>

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static const double FS = 0.08;

// Options with trimming off: the segmenter before trim_sec existed.
static SegmenterOpts legacy(SegmenterOpts o = SegmenterOpts()) {
    o.trim_sec = 0.0;
    return o;
}

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
    CHECK(near(o.trim_sec, 0.3));
    CHECK(near(default_segmenter_opts(VadKind::kHead).trim_sec, 0.3));
    CHECK(near(default_segmenter_opts(VadKind::kSilero).trim_sec, 0.3));
}

static void test_short_is_single() {
    SegmenterOpts o = legacy();
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
    SegmenterOpts o = legacy();
    // 70 s of p == 0.5 exactly is all speech: hard cuts, nothing dropped.
    std::vector<float> p(875, 0.5f);
    auto s = segment_by_vad(p, 70.0, o);
    CHECK(s.size() == 3);
    // Just below the threshold is silence: every segment is dropped.
    std::vector<float> q(875, 0.4999f);
    CHECK(segment_by_vad(q, 70.0, o).empty());
}

static void test_all_speech_hard_cuts() {
    SegmenterOpts o = legacy();
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
    SegmenterOpts o = legacy();
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
    SegmenterOpts o = legacy();
    const double total = 40.0;
    // A long pause at 10-12 s and a short one (0.4 s) at 27 s. The last one wins.
    auto p = make_p((int)std::ceil(total / FS), {{125, 150}, {337, 342}});
    auto s = segment_by_vad(p, total, o);
    check_ordered(s, total, o.max_seg_sec);
    CHECK(s.size() == 2);
    if (s.size() == 2) CHECK(near(s[0].end, 339 * FS));  // (337 + 342) / 2
}

static void test_pause_must_be_fully_inside() {
    SegmenterOpts o = legacy();
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
    SegmenterOpts o = legacy();
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
    SegmenterOpts o = legacy();
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
    SegmenterOpts o = legacy();
    const double total = 45.0;
    // 2 frames (0.16 s) is shorter than 0.2 s: hard cut at 30 s.
    auto s = segment_by_vad(make_p((int)std::ceil(total / FS), {{312, 314}}), total, o);
    CHECK(s.size() == 2 && near(s[0].end, 30.0));
    // 3 frames (0.24 s) is long enough.
    s = segment_by_vad(make_p((int)std::ceil(total / FS), {{312, 315}}), total, o);
    CHECK(s.size() == 2 && near(s[0].end, 313 * FS));
}

static void test_bridge_short_speech_gap() {
    SegmenterOpts o = legacy();
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
    SegmenterOpts o = legacy();
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
    SegmenterOpts o = legacy();
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
    SegmenterOpts o = legacy();
    const double total = 65.0;
    CHECK(segment_by_vad(make_p((int)std::ceil(total / FS), {{0, 100000}}), total, o).empty());
    CHECK(segment_by_vad({}, 65.0, o).empty());
}

static void test_random_property() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uniform_int_distribution<int> run_len(1, 12);
    SegmenterOpts o = legacy();
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
        SegmenterOpts o = legacy();
        o.frame_sec = fs;
        auto s = segment_by_vad(std::vector<float>(100, 0.0f), 100.0, o);
        CHECK(s.size() == 1);
        if (s.size() == 1) CHECK(s[0].start == 0.0 && s[0].end == 100.0);
    }
    SegmenterOpts o = legacy();
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
            SegmenterOpts o = legacy();
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
        SegmenterOpts o = legacy();
        o.threshold = t;
        auto s = segment_by_vad(std::vector<float>(2000, 0.0f), 100.0, o);
        CHECK(s.size() == 1);
        if (s.size() == 1) CHECK(s[0].start == 0.0 && s[0].end == 100.0);
    }
}

// speech_regions: smoothed speech runs for audio of any length, with gaps
// shorter than min_pause merged.
static void test_speech_regions() {
    SegmenterOpts o = legacy();
    // 50 frames (4 s): silence 0-10, speech 10-20, silence 20-23 (0.24 s, a
    // pause), speech 23-35, silence 35-50.
    std::vector<float> p(50, 0.02f);
    for (int i = 10; i < 20; ++i) p[(size_t)i] = 0.95f;
    for (int i = 23; i < 35; ++i) p[(size_t)i] = 0.95f;
    auto r = speech_regions(p, 4.0, o);  // short audio still gives regions
    CHECK(r.size() == 2);
    if (r.size() == 2) {
        CHECK(near(r[0].start, 0.8) && near(r[0].end, 1.6));
        CHECK(near(r[1].start, 1.84) && near(r[1].end, 2.8));
    }
    // A longer min_pause merges the two regions (the gap is 0.24 s).
    o.min_pause_sec = 0.3;
    r = speech_regions(p, 4.0, o);
    CHECK(r.size() == 1);
    if (r.size() == 1) CHECK(near(r[0].start, 0.8) && near(r[0].end, 2.8));
    // Runs shorter than min_speech are dropped; the threshold is inclusive.
    o = SegmenterOpts();
    std::vector<float> q(30, 0.02f);
    q[5] = 0.95f;                                   // 80 ms: below min_speech 0.1 s
    for (int i = 10; i < 14; ++i) q[(size_t)i] = 0.5f;  // exactly the threshold
    r = speech_regions(q, 2.4, o);
    CHECK(r.size() == 1);
    if (r.size() == 1) CHECK(near(r[0].start, 0.8) && near(r[0].end, 1.12));
    // Speech up to the end is clipped to total_sec.
    std::vector<float> e(10, 0.95f);
    r = speech_regions(e, 0.75, o);
    CHECK(r.size() == 1 && near(r[0].start, 0.0) && near(r[0].end, 0.75));
    // No speech, no probabilities, higher threshold.
    CHECK(speech_regions(std::vector<float>(10, 0.02f), 0.8, o).empty());
    CHECK(speech_regions({}, 0.0, o).empty());
    o.threshold = 0.99f;
    CHECK(speech_regions(e, 0.8, o).empty());
    // Long audio is not cut at max_seg_sec.
    std::vector<float> l(1000, 0.95f);
    r = speech_regions(l, 80.0, SegmenterOpts());
    CHECK(r.size() == 1 && near(r[0].end, 80.0));
    // Degenerate options give the whole clip, like segment_by_vad.
    SegmenterOpts bad;
    bad.frame_sec = 0.0;
    r = speech_regions(e, 0.8, bad);
    CHECK(r.size() == 1 && near(r[0].end, 0.8));
}


// ---- Silero (32 ms frames) ----------------------------------------------

static void test_kind_defaults() {
    const SegmenterOpts h = default_segmenter_opts(VadKind::kHead);
    const SegmenterOpts d;
    CHECK(near(h.frame_sec, 0.08) && h.threshold == d.threshold && near(h.min_pause_sec, d.min_pause_sec) &&
          near(h.min_speech_sec, d.min_speech_sec) && near(h.bridge_sec, d.bridge_sec) &&
          near(h.max_seg_sec, d.max_seg_sec) && near(h.min_seg_sec, d.min_seg_sec) && h.pad_sec == 0.0);
    const SegmenterOpts s = default_segmenter_opts(VadKind::kSilero);
    CHECK(near(s.frame_sec, 0.032) && s.threshold == 0.5f);
    CHECK(near(s.min_speech_sec, 0.25) && near(s.min_pause_sec, 0.1) && near(s.bridge_sec, 0.1) && near(s.pad_sec, 0.03));
    CHECK(near(s.max_seg_sec, 30.0) && near(s.min_seg_sec, 1.0));
}

// Frames of 32 ms: boundaries are whole frames, a pause is found, caps hold.
static void test_segmenter_32ms() {
    SegmenterOpts o = legacy(default_segmenter_opts(VadKind::kSilero));
    const double fs = 0.032;
    // 50 s: speech, a 0.5 s pause at 20 s, speech to the end.
    const int n = (int)std::ceil(50.0 / fs);
    std::vector<float> p((size_t)n, 0.9f);
    const int a = (int)std::llround(20.0 / fs), b = a + 16;
    for (int i = a; i < b; ++i) p[(size_t)i] = 0.01f;
    const auto s = segment_by_vad(p, 50.0, o);
    CHECK(s.size() == 2);
    if (s.size() == 2) {
        CHECK(near(s[0].start, 0.0) && near(s[1].end, 50.0));
        CHECK(near(s[0].end, s[1].start));
        const double mid = (double)((a + b) / 2) * fs;
        CHECK(near(s[0].end, mid));
        const double q = s[0].end / fs;
        CHECK(std::fabs(q - std::round(q)) < 1e-6);
    }
    check_ordered(s, 50.0, o.max_seg_sec);
    // All speech: hard cuts at max_seg_sec rounded down to whole frames (937 frames).
    const auto h = segment_by_vad(std::vector<float>((size_t)n, 0.9f), 50.0, o);
    CHECK(h.size() == 2 && near(h[0].end, 937 * fs));
    // The same options at 80 ms give the head's result: frame_sec only sets the grid.
    SegmenterOpts o8;
    const auto r8 = segment_by_vad(make_p(750, {{300, 310}}), 60.0, o8);
    SegmenterOpts o8b = default_segmenter_opts(VadKind::kHead);
    const auto r8b = segment_by_vad(make_p(750, {{300, 310}}), 60.0, o8b);
    CHECK(r8.size() == r8b.size());
    for (size_t i = 0; i < r8.size() && i < r8b.size(); ++i) CHECK(r8[i].start == r8b[i].start && r8[i].end == r8b[i].end);
}

static void test_silero_speech_regions_and_pad() {
    SegmenterOpts o = default_segmenter_opts(VadKind::kSilero);
    const double fs = 0.032;
    // 3 s: speech frames [20, 60) = 1.28 s, a 0.256 s pause, speech [68, 80) = 0.384 s.
    std::vector<float> p(94, 0.01f);
    for (int i = 20; i < 60; ++i) p[(size_t)i] = 0.9f;
    for (int i = 68; i < 80; ++i) p[(size_t)i] = 0.9f;
    auto r = speech_regions(p, 3.0, o);
    CHECK(r.size() == 2);
    if (r.size() == 2) {
        CHECK(near(r[0].start, 20 * fs - 0.03) && near(r[0].end, 60 * fs + 0.03));
        CHECK(near(r[1].start, 68 * fs - 0.03) && near(r[1].end, 80 * fs + 0.03));
    }
    // A run shorter than 250 ms (7 frames = 224 ms) is dropped, 8 frames (256 ms) kept.
    std::vector<float> q(60, 0.01f);
    for (int i = 5; i < 12; ++i) q[(size_t)i] = 0.9f;
    for (int i = 30; i < 38; ++i) q[(size_t)i] = 0.9f;
    r = speech_regions(q, 60 * fs, o);
    CHECK(r.size() == 1 && near(r[0].start, 30 * fs - 0.03));
    // A gap shorter than 100 ms (3 frames) is bridged, 4 frames (128 ms) is not.
    std::vector<float> g(60, 0.01f);
    for (int i = 5; i < 20; ++i) g[(size_t)i] = 0.9f;
    for (int i = 23; i < 40; ++i) g[(size_t)i] = 0.9f;  // gap of 3 frames
    CHECK(speech_regions(g, 60 * fs, o).size() == 1);
    for (int i = 20; i < 24; ++i) g[(size_t)i] = 0.01f; // gap of 4 frames
    CHECK(speech_regions(g, 60 * fs, o).size() == 2);
    // Pad is clamped to [0, total] and neighbours meet in the middle of a small gap.
    std::vector<float> e(20, 0.9f);
    r = speech_regions(e, 20 * fs, o);
    CHECK(r.size() == 1 && near(r[0].start, 0.0) && near(r[0].end, 20 * fs));
    SegmenterOpts w = o;
    w.pad_sec = 0.2;   // gap of 4 frames = 0.128 s < 2 * pad
    r = speech_regions(g, 60 * fs, w);
    CHECK(r.size() == 2);
    if (r.size() == 2) CHECK(near(r[0].end, r[1].start) && near(r[0].end, 20 * fs + 0.064));
    // No pad keeps the plain regions, and a bad pad is degenerate.
    w.pad_sec = 0.0;
    r = speech_regions(g, 60 * fs, w);
    CHECK(r.size() == 2 && near(r[0].start, 5 * fs) && near(r[0].end, 20 * fs));
    w.pad_sec = -1.0;
    r = speech_regions(g, 60 * fs, w);
    CHECK(r.size() == 1 && near(r[0].start, 0.0) && near(r[0].end, 60 * fs));
}

// The online tracker gives the regions of speech_regions on random input.
static void test_event_tracker_matches_offline() {
    std::mt19937 rng(7);
    int cases = 0;
    for (int kind = 0; kind < 2; ++kind) {
        for (int iter = 0; iter < 400; ++iter) {
            SegmenterOpts o = default_segmenter_opts(kind ? VadKind::kSilero : VadKind::kHead);
            if (iter % 3 == 1) o.min_pause_sec = 0.05 + 0.01 * (double)(rng() % 40);
            if (iter % 5 == 2) o.min_speech_sec = 0.02 + 0.01 * (double)(rng() % 60);
            if (iter % 4 == 3) o.bridge_sec = 0.01 * (double)(rng() % 30);
            if (iter % 2 == 0) o.pad_sec = std::min(o.min_pause_sec / 2.0, 0.01 * (double)(rng() % 8));
            const int n = 1 + (int)(rng() % 300);
            // Runs of random length with random value.
            std::vector<float> p;
            bool sp = rng() & 1;
            while ((int)p.size() < n) {
                const int len = 1 + (int)(rng() % 14);
                for (int i = 0; i < len && (int)p.size() < n; ++i) p.push_back(sp ? 0.9f : 0.05f);
                sp = !sp;
            }
            const double total = (double)n * o.frame_sec - (double)(rng() % 100) / 100.0 * o.frame_sec * 0.9;
            const auto want = speech_regions(p, total, o);
            VadEventTracker t(o);
            std::vector<VadEvent> ev;
            for (float v : p) t.push(v, &ev);
            t.finish(total, &ev);
            CHECK(ev.size() == want.size() * 2);
            bool ok = ev.size() == want.size() * 2;
            for (size_t i = 0; ok && i < want.size(); ++i) {
                ok = ev[2 * i].start && !ev[2 * i + 1].start && near(ev[2 * i].time, want[i].start) &&
                     near(ev[2 * i + 1].time, want[i].end);
            }
            if (!ok) std::fprintf(stderr, "tracker mismatch kind %d iter %d (%zu events, %zu regions) pause %.3f speech %.3f bridge %.3f pad %.3f\n", kind, iter, ev.size(), want.size(), o.min_pause_sec, o.min_speech_sec, o.bridge_sec, o.pad_sec);
            CHECK(ok);
            ++cases;
        }
    }
    CHECK(cases == 800);
    // Events arrive as they become known: the end of a region comes only after
    // min_pause of silence, a start only after min_speech of speech.
    SegmenterOpts o = default_segmenter_opts(VadKind::kSilero);
    VadEventTracker t(o);
    std::vector<VadEvent> ev;
    for (int i = 0; i < 6; ++i) t.push(0.9f, &ev);   // 192 ms < 250 ms
    CHECK(ev.empty());
    t.push(0.9f, &ev); t.push(0.9f, &ev);            // 256 ms
    CHECK(ev.size() == 1 && ev[0].start && near(ev[0].time, 0.0));
    for (int i = 0; i < 3; ++i) t.push(0.01f, &ev);  // 96 ms: not yet a pause
    CHECK(ev.size() == 1);
    t.push(0.01f, &ev);                              // 128 ms
    CHECK(ev.size() == 2 && !ev[1].start && near(ev[1].time, 8 * 0.032 + 0.03));
    // reset() starts a fresh stream; the degenerate option set gives one region.
    t.reset();
    ev.clear();
    CHECK(t.frames() == 0);
    SegmenterOpts bad;
    bad.frame_sec = 0.0;
    VadEventTracker tb(bad);
    tb.push(0.9f, &ev); tb.push(0.0f, &ev); tb.finish(0.5, &ev);
    CHECK(ev.size() == 2 && ev[0].start && near(ev[1].time, 0.5));
}


// ---- trim_sec ------------------------------------------------------------

// n frames at p = 0.02 with speech (p = 0.95) in the given frame ranges.
static std::vector<float> speech_p(int n, std::initializer_list<std::pair<int, int>> speech) {
    std::vector<float> p((size_t)n, 0.02f);
    for (auto r : speech)
        for (int i = r.first; i < r.second && i < n; ++i) p[(size_t)i] = 0.95f;
    return p;
}

static bool seg_is(const std::vector<VadSegment>& s, std::initializer_list<std::pair<double, double>> want) {
    if (s.size() != want.size()) return false;
    size_t i = 0;
    for (auto w : want) {
        if (!near(s[i].start, w.first) || !near(s[i].end, w.second)) return false;
        ++i;
    }
    return true;
}

// Head frames (0.08 s), default trim 0.3 s.
static void test_trim_head() {
    const SegmenterOpts o;  // trim_sec 0.3
    // Speech [330,375) = 26.4-30.0 s and [450,500) = 36.0-40.0 s in 45 s. The
    // cuts are at the middle of the leading silence (frame 165, which holds no
    // speech, so that piece is dropped) and of the pause (frame 412 = 32.96 s).
    auto p = speech_p(563, {{330, 375}, {450, 500}});
    auto s = segment_by_vad(p, 45.0, o);
    CHECK(seg_is(s, {{26.4 - 0.3, 30.0 + 0.3}, {36.0 - 0.3, 40.0 + 0.3}}));
    // trim 0 keeps the whole cuts.
    s = segment_by_vad(p, 45.0, legacy());
    CHECK(seg_is(s, {{165 * FS, 412 * FS}, {412 * FS, 45.0}}));
    // Speech at both edges of the audio: the start cannot go below 0 and the
    // end not beyond the audio. Cut at the middle of the pause (frame 250).
    p = speech_p(500, {{0, 10}, {490, 500}});
    s = segment_by_vad(p, 40.0, o);
    CHECK(seg_is(s, {{0.0, 0.8 + 0.3}, {39.2 - 0.3, 40.0}}));
    // No speech anywhere: nothing is left, and no segment is trimmed to nothing.
    CHECK(segment_by_vad(speech_p(875, {}), 70.0, o).empty());
    // Two speech runs 0.16 s apart (not a pause) stay in one segment, and the
    // gap between them is kept. The audio after the last run is dropped.
    p = speech_p(563, {{100, 120}, {122, 140}});
    s = segment_by_vad(p, 45.0, o);
    CHECK(seg_is(s, {{8.0 - 0.3, 11.2 + 0.3}}));
    // Two runs far apart, in two segments: each is trimmed on its own, and the
    // trim never reaches into the neighbouring segment.
    p = speech_p(563, {{20, 40}, {60, 80}});
    s = segment_by_vad(p, 45.0, o);
    CHECK(seg_is(s, {{1.6 - 0.3, 3.2 + 0.3}, {4.8 - 0.3, 6.4 + 0.3}}));
    // A trim larger than the pause is limited by the cut (cut at frame 50 = 4.0 s).
    SegmenterOpts big;
    big.trim_sec = 5.0;
    s = segment_by_vad(p, 45.0, big);
    CHECK(s.size() == 2 && near(s[0].start, 0.0) && near(s[0].end, 4.0) && near(s[1].start, 4.0));
    check_ordered(s, 45.0, 30.0);
}

// Silero frames (0.032 s).
static void test_trim_silero() {
    const SegmenterOpts o = default_segmenter_opts(VadKind::kSilero);
    CHECK(near(o.trim_sec, 0.3));
    const double fs = 0.032;
    // Speech at the end of a window and a later run. 45 s, n = 1407.
    //   speech [900,937) and [1200,1250). The first cut is in the leading
    //   silence (frame 450), the second in the pause (frame 1068).
    auto p = speech_p(1407, {{900, 937}, {1200, 1250}});
    auto s = segment_by_vad(p, 45.0, o);
    CHECK(seg_is(s, {{900 * fs - 0.3, 937 * fs + 0.3}, {1200 * fs - 0.3, 1250 * fs + 0.3}}));
    s = segment_by_vad(p, 45.0, legacy(o));
    CHECK(seg_is(s, {{450 * fs, 1068 * fs}, {1068 * fs, 45.0}}));
    // Speech at both edges. 40 s, n = 1250, cut at frame 625 (20 s).
    p = speech_p(1250, {{0, 40}, {1210, 1250}});
    s = segment_by_vad(p, 40.0, o);
    CHECK(seg_is(s, {{0.0, 40 * fs + 0.3}, {1210 * fs - 0.3, 40.0}}));
    // No speech.
    CHECK(segment_by_vad(speech_p(1407, {}), 45.0, o).empty());
    // A short run (below min_speech 0.25 s) is no speech and is not kept.
    CHECK(segment_by_vad(speech_p(1407, {{500, 507}}), 45.0, o).empty());
}

// Speech that runs through a hard cut is not trimmed at the cut, and a clip of
// at most max_seg_sec is returned whole.
static void test_trim_keeps_cuts_and_short_audio() {
    const SegmenterOpts o;
    auto s = segment_by_vad(speech_p(875, {{0, 875}}), 70.0, o);
    CHECK(seg_is(s, {{0.0, 30.0}, {30.0, 60.0}, {60.0, 70.0}}));
    s = segment_by_vad(speech_p(300, {{100, 110}}), 24.0, o);  // 24 s: whole
    CHECK(seg_is(s, {{0.0, 24.0}}));
    s = segment_by_vad(speech_p(375, {{100, 110}}), 30.0, o);  // exactly the cap
    CHECK(seg_is(s, {{0.0, 30.0}}));
    // Speech only in the leading 0.1 s of the second half of a hard cut.
    s = segment_by_vad(speech_p(875, {{0, 375}, {375, 377}}), 70.0, o);
    CHECK(s.size() == 2 && near(s[1].start, 30.0) && near(s[1].end, 30.0 + 0.16 + 0.3));
    // A bad trim is degenerate like the other options.
    SegmenterOpts bad;
    bad.trim_sec = NAN;
    s = segment_by_vad(speech_p(875, {{0, 10}}), 70.0, bad);
    CHECK(seg_is(s, {{0.0, 70.0}}));
    // A negative trim means off.
    bad.trim_sec = -1.0;
    s = segment_by_vad(speech_p(563, {{20, 40}, {60, 80}}), 45.0, bad);
    CHECK(s.size() == 2 && near(s[0].start, 0.0) && near(s[0].end, 4.0));
}

// Random streams: trimmed segments are inside the old cuts, contain the same
// speech, and never exceed trim_sec of non-speech at either edge.
static void test_trim_random_property() {
    std::mt19937 rng(11);
    for (int trial = 0; trial < 200; ++trial) {
        const SegmenterOpts old = legacy(default_segmenter_opts(trial % 2 ? VadKind::kSilero : VadKind::kHead));
        SegmenterOpts nw = old;
        nw.trim_sec = 0.1 * (double)(rng() % 10);
        const double fs = old.frame_sec;
        const int n = (int)(35.0 / fs) + (int)(rng() % 2000);
        std::vector<float> p;
        bool sp = rng() & 1;
        while ((int)p.size() < n) {
            const int len = 1 + (int)(rng() % 300);
            for (int i = 0; i < len && (int)p.size() < n; ++i) p.push_back(sp ? 0.9f : 0.05f);
            sp = !sp;
        }
        const double total = (double)n * fs;
        const auto a = segment_by_vad(p, total, old);
        const auto b = segment_by_vad(p, total, nw);
        CHECK(a.size() == b.size());
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
            CHECK(b[i].start >= a[i].start - 1e-9 && b[i].end <= a[i].end + 1e-9);
            CHECK(b[i].end > b[i].start);
        }
        check_ordered(b, total, old.max_seg_sec);
    }
}

// With trim_sec 0 the output equals what the segmenter gave before trimming
// existed: the digest was computed with the previous implementation on the same
// seeded streams (80 ms and 32 ms frames, assorted options).
// Digest of segment_by_vad over seeded random inputs (FNV-1a over the printed segments).
static uint64_t digest_segments() {
    std::mt19937 rng(20261004);
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](const char* s) { for (; *s; ++s) { h ^= (unsigned char)*s; h *= 1099511628211ull; } };
    for (int kind = 0; kind < 2; ++kind) {
        for (int iter = 0; iter < 150; ++iter) {
            SegmenterOpts o = default_segmenter_opts(kind ? VadKind::kSilero : VadKind::kHead);
            o.trim_sec = 0.0;
            if (iter % 3 == 1) o.min_pause_sec = 0.05 + 0.01 * (double)(rng() % 40);
            if (iter % 4 == 2) o.max_seg_sec = 5.0 + (double)(rng() % 250) / 10.0;
            const int n = (int)(o.max_seg_sec / o.frame_sec) + 1 + (int)(rng() % 4000);
            std::vector<float> p;
            bool sp = rng() & 1;
            while ((int)p.size() < n) {
                const int len = 1 + (int)(rng() % (iter % 2 ? 40 : 400));
                for (int i = 0; i < len && (int)p.size() < n; ++i) p.push_back(sp ? 0.9f : 0.05f);
                sp = !sp;
            }
            const double total = (double)n * o.frame_sec - (double)(rng() % 100) / 100.0 * o.frame_sec * 0.9;
            char buf[96];
            std::snprintf(buf, sizeof(buf), "#%d,%d;", kind, iter);
            mix(buf);
            for (const VadSegment& g : segment_by_vad(p, total, o)) {
                std::snprintf(buf, sizeof(buf), "%.9f,%.9f;", g.start, g.end);
                mix(buf);
            }
        }
    }
    return h;
}

static void test_trim_zero_equals_previous_output() {
    CHECK(digest_segments() == 16100934503807512709ull);
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
    test_speech_regions();
    test_random_property();
    test_kind_defaults();
    test_segmenter_32ms();
    test_silero_speech_regions_and_pad();
    test_event_tracker_matches_offline();
    test_trim_head();
    test_trim_silero();
    test_trim_keeps_cuts_and_short_audio();
    test_trim_random_property();
    test_trim_zero_equals_previous_output();
    if (failures) return 1;
    std::puts("test_vad_segmenter: OK");
    return 0;
}
