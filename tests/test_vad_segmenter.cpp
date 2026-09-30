#include "vad_segmenter.hpp"

#include <cmath>
#include <cstdio>
#include <random>
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
    // pauses centered near 25.5 s and 55.5 s
    auto p = make_p((int)std::ceil(total / FS), {{312, 325}, {692, 705}});
    auto s = segment_by_vad(p, total, o);
    check_tiling(s, total, o.max_seg_sec);
    CHECK(s.size() == 3);
    if (s.size() == 3) {
        CHECK(std::fabs(s[0].end - 25.5) < 0.2);
        CHECK(std::fabs(s[1].end - 55.5) < 0.2);
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
    SegmenterOpts o;
    for (int trial = 0; trial < 200; ++trial) {
        const double total = 31.0 + 600.0 * u(rng);
        const int n = (int)std::ceil(total / FS);
        std::vector<float> p((size_t)n);
        for (auto& v : p) v = u(rng) < 0.8f ? 0.9f : 0.05f;
        check_tiling(segment_by_vad(p, total, o), total, o.max_seg_sec);
    }
}

int main() {
    test_short_is_single();
    test_all_speech_hard_cuts();
    test_cuts_land_in_pauses();
    test_picks_longest_pause();
    test_all_silence();
    test_pause_shorter_than_min_is_ignored();
    test_random_property();
    if (failures) return 1;
    std::puts("test_vad_segmenter: OK");
    return 0;
}
