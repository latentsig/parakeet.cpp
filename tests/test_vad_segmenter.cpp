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

static void check_tiling(const std::vector<VadSegment>& s, double total, double max_seg) {
    CHECK(!s.empty());
    if (s.empty()) return;
    CHECK(s.front().start == 0.0);
    CHECK(std::fabs(s.back().end - total) < 1e-9);
    for (size_t i = 0; i < s.size(); ++i) {
        CHECK(s[i].end > s[i].start);
        CHECK(s[i].end - s[i].start <= max_seg + 1e-9);
        if (i + 1 < s.size()) CHECK(s[i].end == s[i + 1].start);
    }
}

static void test_short_is_single() {
    SegmenterOpts o;
    auto s = segment_by_vad(make_p(250, {}), 20.0, o);
    CHECK(s.size() == 1 && s[0].start == 0.0 && s[0].end == 20.0);
    s = segment_by_vad({}, 0.5, o);            // shorter than one frame
    CHECK(s.size() == 1 && s[0].end == 0.5);
    s = segment_by_vad(make_p(375, {}), 30.0, o);  // exactly the cap
    CHECK(s.size() == 1 && s[0].end == 30.0);
}

static void test_all_speech_hard_cuts() {
    SegmenterOpts o;
    const double total = 70.0;
    auto s = segment_by_vad(make_p((int)std::ceil(total / FS), {}), total, o);
    check_tiling(s, total, o.max_seg_sec);
    CHECK(s.size() == 3);
}

static void test_cuts_land_in_pauses() {
    SegmenterOpts o;
    const double total = 70.0;
    // pauses at [312,325) and [640,653): midpoints 318 (25.44 s) and 646 (51.68 s)
    auto p = make_p((int)std::ceil(total / FS), {{312, 325}, {640, 653}});
    auto s = segment_by_vad(p, total, o);
    check_tiling(s, total, o.max_seg_sec);
    CHECK(s.size() == 3);
    if (s.size() == 3) {
        CHECK(std::fabs(s[0].end - 25.44) < 0.1);
        CHECK(std::fabs(s[1].end - 51.68) < 0.1);
    }
}

static void test_picks_longest_pause() {
    SegmenterOpts o;
    const double total = 40.0;
    // two pauses in the last third of the first window: 22-22.4 s (short) and 27-28 s (long)
    auto p = make_p((int)std::ceil(total / FS), {{275, 280}, {337, 350}});
    auto s = segment_by_vad(p, total, o);
    check_tiling(s, total, o.max_seg_sec);
    CHECK(s.size() == 2);
    if (s.size() == 2) CHECK(std::fabs(s[0].end - 27.5) < 0.3);
}

static void test_all_silence() {
    SegmenterOpts o;
    const double total = 65.0;
    auto s = segment_by_vad(make_p((int)std::ceil(total / FS), {{0, 100000}}), total, o);
    check_tiling(s, total, o.max_seg_sec);
}

static void test_pause_shorter_than_min_is_ignored() {
    SegmenterOpts o;
    const double total = 45.0;
    // a 2-frame (0.16 s) pause at 25 s is below min_pause_sec (0.32 s): hard cut at 30 s
    auto s = segment_by_vad(make_p((int)std::ceil(total / FS), {{312, 314}}), total, o);
    check_tiling(s, total, o.max_seg_sec);
    CHECK(s.size() == 2 && std::fabs(s[0].end - 30.0) < 0.1);
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
        // Generate bursty silence: runs of 1 to 12 frames
        for (int i = 0; i < n; ) {
            if (u(rng) < 0.2f) {  // start a silence run
                int len = run_len(rng);
                for (int j = 0; j < len && i + j < n; ++j) p[(size_t)(i + j)] = 0.05f;
                i += len;
            } else {
                ++i;
            }
        }
        check_tiling(segment_by_vad(p, total, o), total, o.max_seg_sec);
    }
}

static void test_fallback_window() {
    // Pause at frames [150,163) only (in first window but not in last third).
    // First search [250,375) finds nothing; fallback search [100,375) finds [150,163).
    // Midpoint 156 => 12.48 s. Total 42.4 s (531 frames) yields exactly 2 segments.
    SegmenterOpts o;
    const double total = 42.4;
    auto p = make_p((int)std::ceil(total / FS), {{150, 163}});
    auto s = segment_by_vad(p, total, o);
    check_tiling(s, total, o.max_seg_sec);
    CHECK(s.size() == 2);
    if (s.size() == 2) CHECK(std::fabs(s[0].end - 12.48) < 0.1);
}

static void test_min_seg_respected() {
    // Pause at frames [50,63) (4.0 to 5.04 s) only.
    // Fallback window starts at min_seg frame 100, so pause is ignored.
    // Expected hard cut at 30.0 s.
    SegmenterOpts o;
    const double total = 45.0;
    auto p = make_p((int)std::ceil(total / FS), {{50, 63}});
    auto s = segment_by_vad(p, total, o);
    check_tiling(s, total, o.max_seg_sec);
    CHECK(s.size() == 2 && std::fabs(s[0].end - 30.0) < 0.1);
}

static void test_tie_goes_to_later_run() {
    // Two equal 13-frame pauses at [270,283) and [330,343), both in last third.
    // When tied, later run wins. Midpoint of [330,343) is 336 => 26.88 s.
    SegmenterOpts o;
    const double total = 45.0;
    auto p = make_p((int)std::ceil(total / FS), {{270, 283}, {330, 343}});
    auto s = segment_by_vad(p, total, o);
    check_tiling(s, total, o.max_seg_sec);
    CHECK(s.size() == 2);
    if (s.size() == 2) CHECK(std::fabs(s[0].end - 26.88) < 0.1);
}

static void test_exact_totals() {
    // Test 1: total 60.0 s (no pauses, all speech).
    // Expected: exactly two segments [0, 30] and [30, 60].
    SegmenterOpts o;
    auto s = segment_by_vad(make_p(750, {}), 60.0, o);
    check_tiling(s, 60.0, o.max_seg_sec);
    CHECK(s.size() == 2);
    if (s.size() == 2) {
        CHECK(std::fabs(s[0].end - 30.0) < 1e-6);
        CHECK(std::fabs(s[1].end - 60.0) < 1e-6);
    }

    // Test 2: total 30.08 s with 376 frames (no pauses).
    // Expected: two segments with s[0].end at 30.0 and last segment about 0.08 s.
    s = segment_by_vad(make_p(376, {}), 30.08, o);
    check_tiling(s, 30.08, o.max_seg_sec);
    CHECK(s.size() == 2);
    if (s.size() == 2) {
        CHECK(std::fabs(s[0].end - 30.0) < 1e-6);
        CHECK(std::fabs((s[1].end - s[1].start) - 0.08) < 1e-6);
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

int main() {
    test_degenerate_opts();
    test_short_is_single();
    test_all_speech_hard_cuts();
    test_cuts_land_in_pauses();
    test_picks_longest_pause();
    test_all_silence();
    test_pause_shorter_than_min_is_ignored();
    test_fallback_window();
    test_min_seg_respected();
    test_tie_goes_to_later_run();
    test_exact_totals();
    test_random_property();
    if (failures) return 1;
    std::puts("test_vad_segmenter: OK");
    return 0;
}
