// The opt-in run gate of the segmenter (SegmenterOpts::run_gate): a speech run
// is dropped when the median of its frame probabilities is below the gate.
// Model-independent: synthetic probability streams with exact expected output,
// for the 0.08 s head frame and the 0.032 s Silero frame.
//
// The definition under test:
//   * a run is a maximal stretch of frames with p >= threshold (before any
//     bridging), so a bridged gap is never inside a run;
//   * the median is over the probabilities of the frames of that run (odd
//     count: the middle value; even count: the mean of the two middle values);
//   * the run is kept when median >= run_gate, dropped when it is below;
//   * dropped frames count as silence for bridging, the min_speech rule, pauses
//     and trim (the gate runs first);
//   * run_gate 0 is off and gives the output of the segmenter before the gate.
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

static bool near(double a, double b) { return std::fabs(a - b) < 1e-6; }

struct Fill { int a, b; float v; };  // frames [a, b) at probability v

// n frames at p = 0.02, then the fills in order (a later fill overwrites).
static std::vector<float> stream(int n, std::initializer_list<Fill> fills) {
    std::vector<float> p((size_t)n, 0.02f);
    for (const Fill& f : fills)
        for (int i = f.a; i < f.b && i < n; ++i) p[(size_t)i] = f.v;
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

static bool same_segs(const std::vector<VadSegment>& a, const std::vector<VadSegment>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].start != b[i].start || a[i].end != b[i].end) return false;  // exact: byte for byte
    return true;
}

static SegmenterOpts gated(SegmenterOpts o, float g) {
    o.run_gate = g;
    return o;
}

static void test_default_is_off() {
    CHECK(SegmenterOpts().run_gate == 0.0f);
    CHECK(default_segmenter_opts(VadKind::kHead).run_gate == 0.0f);
    CHECK(default_segmenter_opts(VadKind::kSilero).run_gate == 0.0f);
}

// Head frames. Runs: A [10,30) at 0.97, B [50,70) at 0.80, C [100,110) at 0.6
// with a single peak of 0.99 (low median, high peak), D one frame at 0.99.
static void test_head_speech_regions() {
    SegmenterOpts o;
    o.min_speech_sec = 0.05;  // a one-frame run (0.08 s) is speech
    const double total = 15.0;
    auto p = stream(188, {{10, 30, 0.97f}, {50, 70, 0.80f}, {100, 110, 0.6f}, {105, 106, 0.99f}, {130, 131, 0.99f}});
    // Off: all four runs.
    CHECK(seg_is(speech_regions(p, total, o), {{0.8, 2.4}, {4.0, 5.6}, {8.0, 8.8}, {10.4, 10.48}}));
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.0f)), {{0.8, 2.4}, {4.0, 5.6}, {8.0, 8.8}, {10.4, 10.48}}));
    // 0.7: C goes although its peak is 0.99 (its median is 0.6); B stays.
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.7f)), {{0.8, 2.4}, {4.0, 5.6}, {10.4, 10.48}}));
    // 0.9: B goes as well. The single frame at 0.99 has median 0.99 and stays.
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.9f)), {{0.8, 2.4}, {10.4, 10.48}}));
    // 0.995: nothing is left.
    CHECK(speech_regions(p, total, gated(o, 0.995f)).empty());
}

// Boundary: median exactly equal to the gate keeps the run (>=).
static void test_boundary() {
    SegmenterOpts o;
    const double total = 8.0;
    // Odd run: 3 frames at 0.75 (exact in float): median 0.75.
    auto p = stream(100, {{10, 13, 0.75f}});
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.75f)), {{0.8, 1.04}}));
    CHECK(speech_regions(p, total, gated(o, 0.76f)).empty());
    // Even run, values not in order: {0.875, 0.5, 1.0, 0.625} has median
    // (0.625 + 0.875) / 2 = 0.75 and mean 0.75.
    p = stream(100, {{10, 11, 0.875f}, {11, 12, 0.5f}, {12, 13, 1.0f}, {13, 14, 0.625f}});
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.75f)), {{0.8, 1.12}}));
    CHECK(speech_regions(p, total, gated(o, 0.76f)).empty());
    // {0.5, 0.5, 0.875, 1.0}: median 0.6875, mean 0.719. The median decides.
    p = stream(100, {{10, 12, 0.5f}, {12, 13, 0.875f}, {13, 14, 1.0f}});
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.6875f)), {{0.8, 1.12}}));
    CHECK(speech_regions(p, total, gated(o, 0.7f)).empty());
    // A frame exactly at the threshold belongs to the run: {0.5, 0.5, 0.5}.
    p = stream(100, {{10, 13, 0.5f}});
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.5f)), {{0.8, 1.04}}));
    CHECK(speech_regions(p, total, gated(o, 0.51f)).empty());
}

// The run is the raw run of p >= threshold, not the bridged one.
static void test_bridged_gap() {
    SegmenterOpts o;  // bridge 0.1 s bridges a one-frame gap (0.08 s) at 0.08 s frames
    const double total = 8.0;
    // Two good runs and a one-frame gap: bridged into one region, gate keeps it.
    auto p = stream(100, {{10, 20, 0.97f}, {21, 31, 0.97f}});
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.9f)), {{0.8, 2.48}}));
    CHECK(seg_is(speech_regions(p, total, o), {{0.8, 2.48}}));
    // Run 1 is weak (0.6 x 3), run 2 is strong (0.97 x 20), one frame between.
    // The bridged run would have median 0.97 over its 24 frames; the raw run 1
    // has median 0.6 and goes. Only run 2 is left.
    p = stream(100, {{10, 13, 0.6f}, {14, 34, 0.97f}});
    CHECK(seg_is(speech_regions(p, total, o), {{0.8, 2.72}}));
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.9f)), {{1.12, 2.72}}));
    // Run 1 strong, run 2 weak after a bridgeable gap: only run 1 stays, the
    // bridged gap does not come back.
    p = stream(100, {{10, 20, 0.97f}, {21, 24, 0.6f}});
    CHECK(seg_is(speech_regions(p, total, o), {{0.8, 1.92}}));
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.9f)), {{0.8, 1.6}}));
    // Dropping a run that sat between two kept runs opens a gap that is no
    // longer bridged: kept 10 frames, weak 1 frame ... gap 1 + weak run 2 + gap 1.
    p = stream(100, {{10, 20, 0.97f}, {21, 23, 0.6f}, {24, 34, 0.97f}});
    CHECK(seg_is(speech_regions(p, total, o), {{0.8, 2.72}}));
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.9f)), {{0.8, 1.6}, {1.92, 2.72}}));
}

// A gated-out run is silence for the min_speech rule too: a short run next to a
// dropped one is not merged with it.
static void test_min_speech_after_gate() {
    SegmenterOpts o;  // min_speech 0.1 s: one frame (0.08 s) is too short, two are enough
    const double total = 8.0;
    auto p = stream(100, {{10, 12, 0.99f}, {20, 21, 0.99f}});
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.9f)), {{0.8, 0.96}}));
    // The single frame is not rescued by the gate; it fails min_speech first or
    // after, with the same result.
    p = stream(100, {{10, 11, 0.99f}});
    CHECK(speech_regions(p, total, gated(o, 0.9f)).empty());
}

// segment_by_vad on 45 s of head frames (563 frames): speech [20,40) at 0.97 and
// a noise-like run at 0.6.
static void test_head_segments_and_trim() {
    const SegmenterOpts o;  // trim 0.3
    auto p = stream(563, {{20, 40, 0.97f}, {60, 80, 0.6f}});
    CHECK(seg_is(segment_by_vad(p, 45.0, o), {{1.6 - 0.3, 3.2 + 0.3}, {4.8 - 0.3, 6.4 + 0.3}}));
    CHECK(seg_is(segment_by_vad(p, 45.0, gated(o, 0.0f)), {{1.6 - 0.3, 3.2 + 0.3}, {4.8 - 0.3, 6.4 + 0.3}}));
    // The gate drops the second run; the cut stays at the middle of the long
    // silence and the piece is trimmed to the one run that is left.
    CHECK(seg_is(segment_by_vad(p, 45.0, gated(o, 0.9f)), {{1.6 - 0.3, 3.2 + 0.3}}));
    // Everything weak: nothing is left.
    CHECK(segment_by_vad(p, 45.0, gated(o, 0.99f)).empty());

    // Gate first, then trim: a weak run 0.16 s after the strong one is inside
    // the trim margin without the gate and does not count with it.
    p = stream(563, {{20, 40, 0.97f}, {42, 46, 0.6f}});
    CHECK(seg_is(segment_by_vad(p, 45.0, o), {{1.6 - 0.3, 46 * 0.08 + 0.3}}));
    CHECK(seg_is(segment_by_vad(p, 45.0, gated(o, 0.9f)), {{1.6 - 0.3, 3.2 + 0.3}}));
    // With trim 0 the cut itself is unchanged by the gate (only the empty piece
    // is dropped): cut at the middle of the silence after the run.
    SegmenterOpts t0 = o;
    t0.trim_sec = 0.0;
    CHECK(seg_is(segment_by_vad(p, 45.0, gated(t0, 0.9f)), {{0.0, 301 * 0.08}}));

    // Audio of at most max_seg_sec is returned whole, with or without the gate.
    p = stream(300, {{20, 40, 0.6f}});
    CHECK(seg_is(segment_by_vad(p, 24.0, gated(o, 0.9f)), {{0.0, 24.0}}));
    // A gated-out hard-cut piece: speech through the hard cut stays one run.
    p = stream(875, {{0, 875, 0.97f}});
    CHECK(seg_is(segment_by_vad(p, 70.0, gated(o, 0.9f)), {{0.0, 30.0}, {30.0, 60.0}, {60.0, 70.0}}));
    p = stream(875, {{0, 875, 0.6f}});
    CHECK(segment_by_vad(p, 70.0, gated(o, 0.9f)).empty());
}

// Silero frames (0.032 s), default Silero options but no padding.
static void test_silero() {
    SegmenterOpts o = default_segmenter_opts(VadKind::kSilero);
    o.pad_sec = 0.0;
    const double fs = 0.032;
    // A [100,140) 0.97, B [200,240) 0.8, C [300,340) 0.6 with one 0.99 peak.
    auto p = stream(500, {{100, 140, 0.97f}, {200, 240, 0.8f}, {300, 340, 0.6f}, {320, 321, 0.99f}});
    const double total = 16.0;
    CHECK(seg_is(speech_regions(p, total, o), {{100 * fs, 140 * fs}, {200 * fs, 240 * fs}, {300 * fs, 340 * fs}}));
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.7f)), {{100 * fs, 140 * fs}, {200 * fs, 240 * fs}}));
    CHECK(seg_is(speech_regions(p, total, gated(o, 0.9f)), {{100 * fs, 140 * fs}}));
    // With Silero's own padding the kept region is padded as before.
    SegmenterOpts padded = default_segmenter_opts(VadKind::kSilero);
    CHECK(seg_is(speech_regions(p, total, gated(padded, 0.9f)), {{100 * fs - 0.03, 140 * fs + 0.03}}));
    // A run just under min_speech (7 frames = 0.224 s) is dropped with or
    // without the gate; one of 8 frames (0.256 s) is kept without it and dropped
    // by it when weak.
    p = stream(500, {{100, 107, 0.99f}, {200, 208, 0.6f}});
    CHECK(seg_is(speech_regions(p, total, o), {{200 * fs, 208 * fs}}));
    CHECK(speech_regions(p, total, gated(o, 0.9f)).empty());

    // Segments, 45 s = 1407 frames: runs [900,937) and [1200,1250). The second
    // is weak. The first cut lands in the leading silence (its piece has no
    // speech), the second in the pause after the strong run.
    p = stream(1407, {{900, 937, 0.97f}, {1200, 1250, 0.8f}});
    const SegmenterOpts so = default_segmenter_opts(VadKind::kSilero);
    CHECK(seg_is(segment_by_vad(p, 45.0, so), {{900 * fs - 0.3, 937 * fs + 0.3}, {1200 * fs - 0.3, 1250 * fs + 0.3}}));
    CHECK(seg_is(segment_by_vad(p, 45.0, gated(so, 0.9f)), {{900 * fs - 0.3, 937 * fs + 0.3}}));
    // The same stream gated at 0.8 keeps both (median 0.8 >= 0.8).
    CHECK(seg_is(segment_by_vad(p, 45.0, gated(so, 0.8f)), {{900 * fs - 0.3, 937 * fs + 0.3}, {1200 * fs - 0.3, 1250 * fs + 0.3}}));
}

static void test_degenerate_gate() {
    SegmenterOpts o;
    auto p = stream(563, {{20, 40, 0.97f}});
    SegmenterOpts bad = o;
    bad.run_gate = NAN;
    CHECK(seg_is(segment_by_vad(p, 45.0, bad), {{0.0, 45.0}}));
    CHECK(seg_is(speech_regions(p, 45.0, bad), {{0.0, 45.0}}));
    bad.run_gate = INFINITY;
    CHECK(seg_is(segment_by_vad(p, 45.0, bad), {{0.0, 45.0}}));
    // A negative gate is off.
    bad.run_gate = -0.5f;
    CHECK(same_segs(segment_by_vad(p, 45.0, bad), segment_by_vad(p, 45.0, o)));
    CHECK(same_segs(speech_regions(p, 45.0, bad), speech_regions(p, 45.0, o)));
}

// Random streams with random probabilities.
static std::vector<float> random_stream(std::mt19937& rng, int n) {
    std::vector<float> p;
    bool sp = rng() & 1;
    while ((int)p.size() < n) {
        const int len = 1 + (int)(rng() % 60);
        for (int i = 0; i < len && (int)p.size() < n; ++i) {
            const float u = (float)(rng() % 1000) / 1000.0f;
            p.push_back(sp ? 0.5f + 0.5f * u : 0.5f * u * 0.99f);
        }
        sp = !sp;
    }
    return p;
}

static void test_random_properties() {
    std::mt19937 rng(20261006);
    for (int trial = 0; trial < 300; ++trial) {
        SegmenterOpts o = default_segmenter_opts(trial % 2 ? VadKind::kSilero : VadKind::kHead);
        o.pad_sec = 0.0;
        o.trim_sec = (trial % 3) * 0.1;
        const int n = (int)(35.0 / o.frame_sec) + (int)(rng() % 3000);
        const auto p = random_stream(rng, n);
        const double total = (double)n * o.frame_sec;
        // A gate no run can fail (every frame of a run is >= 0.5) changes nothing.
        CHECK(same_segs(segment_by_vad(p, total, gated(o, 0.4f)), segment_by_vad(p, total, o)));
        CHECK(same_segs(speech_regions(p, total, gated(o, 0.4f)), speech_regions(p, total, o)));
        CHECK(same_segs(speech_regions(p, total, gated(o, 1e-9f)), speech_regions(p, total, o)));
        // A gate only removes speech: every gated region lies inside a region of
        // the ungated output (padding off).
        const float g = 0.55f + 0.04f * (float)(trial % 10);
        const auto all = speech_regions(p, total, o);
        const auto some = speech_regions(p, total, gated(o, g));
        for (const VadSegment& s : some) {
            bool inside = false;
            for (const VadSegment& a : all)
                if (s.start >= a.start - 1e-9 && s.end <= a.end + 1e-9) inside = true;
            CHECK(inside);
        }
        // A higher gate never keeps more speech.
        double dur_lo = 0.0, dur_hi = 0.0;
        for (const VadSegment& s : speech_regions(p, total, gated(o, g))) dur_lo += s.end - s.start;
        for (const VadSegment& s : speech_regions(p, total, gated(o, g + 0.05f))) dur_hi += s.end - s.start;
        CHECK(dur_hi <= dur_lo + 1e-9);
        // Segments: ordered, disjoint, each inside the audio.
        double prev = 0.0;
        for (const VadSegment& s : segment_by_vad(p, total, gated(o, g))) {
            CHECK(s.start >= prev - 1e-9 && s.end > s.start && s.end <= total + 1e-9);
            prev = s.end;
        }
    }
}

// The event tracker (streaming) has no run median: the gate does not change it.
static void test_event_tracker_ignores_gate() {
    SegmenterOpts o = default_segmenter_opts(VadKind::kSilero);
    SegmenterOpts g = gated(o, 0.95f);
    auto p = stream(500, {{100, 140, 0.6f}, {200, 240, 0.97f}});
    VadEventTracker a(o), b(g);
    std::vector<VadEvent> ea, eb;
    for (float v : p) { a.push(v, &ea); b.push(v, &eb); }
    a.finish(16.0, &ea);
    b.finish(16.0, &eb);
    CHECK(ea.size() == eb.size() && ea.size() == 4);
    for (size_t i = 0; i < ea.size() && i < eb.size(); ++i)
        CHECK(ea[i].start == eb[i].start && ea[i].time == eb[i].time);
}

int main() {
    test_default_is_off();
    test_head_speech_regions();
    test_boundary();
    test_bridged_gap();
    test_min_speech_after_gate();
    test_head_segments_and_trim();
    test_silero();
    test_degenerate_gate();
    test_random_properties();
    test_event_tracker_ignores_gate();
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_vad_run_gate: OK\n");
    return 0;
}
