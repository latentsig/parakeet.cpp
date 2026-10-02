#include "vad_segmenter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>

namespace pk {

namespace {
// Above this many seconds a frame count could overflow int64.
constexpr double kMaxSec = 1e6;
}

std::vector<VadSegment> segment_by_vad(const std::vector<float>& p, double total_sec,
                                       const SegmenterOpts& o) {
    std::vector<VadSegment> out;
    // Degenerate options: no usable frame grid, so do not cut at all.
    if (!(o.frame_sec > 0.0) || !std::isfinite(o.frame_sec) || !std::isfinite(o.max_seg_sec) ||
        !(o.max_seg_sec > 2.0 * o.frame_sec) || !std::isfinite(total_sec) ||
        !std::isfinite(o.threshold) || !std::isfinite(o.min_seg_sec) ||
        !std::isfinite(o.min_pause_sec) || !std::isfinite(o.bridge_sec) ||
        !std::isfinite(o.min_speech_sec) || o.max_seg_sec > kMaxSec || o.min_seg_sec > kMaxSec ||
        o.min_pause_sec > kMaxSec || o.bridge_sec > kMaxSec || o.min_speech_sec > kMaxSec) {
        out.push_back({0.0, total_sec});
        return out;
    }
    if (total_sec <= o.max_seg_sec) {
        out.push_back({0.0, total_sec});
        return out;
    }
    const double fs = o.frame_sec;
    const int64_t n = std::max<int64_t>(0, (int64_t)std::ceil(total_sec / fs - 1e-9));
    const int64_t max_f = std::max<int64_t>(2, (int64_t)std::floor(o.max_seg_sec / fs + 1e-9));
    const int64_t min_f = std::min<int64_t>(max_f - 1, std::max<int64_t>(1, (int64_t)std::ceil(o.min_seg_sec / fs - 1e-9)));
    const int64_t pause_f = std::max<int64_t>(1, (int64_t)std::ceil(o.min_pause_sec / fs - 1e-9));

    // Step 1: speech mask over n frames (frames without a probability are silent).
    std::vector<char> sp((size_t)n, 0);
    for (int64_t f = 0; f < n && f < (int64_t)p.size(); ++f) sp[(size_t)f] = p[(size_t)f] >= o.threshold;
    // Runs of equal value as [begin, end) frame ranges.
    auto for_runs = [&](bool value, auto&& fn) {
        int64_t f = 0;
        while (f < n) {
            if ((sp[(size_t)f] != 0) != value) { ++f; continue; }
            int64_t e = f;
            while (e < n && (sp[(size_t)e] != 0) == value) ++e;
            fn(f, e);
            f = e;
        }
    };
    // Bridge speech gaps shorter than bridge_sec (only gaps between two speech runs).
    for_runs(false, [&](int64_t a, int64_t b) {
        if (a > 0 && b < n && (double)(b - a) * fs + 1e-9 < o.bridge_sec)
            std::fill(sp.begin() + a, sp.begin() + b, 1);
    });
    // Drop speech runs shorter than min_speech_sec.
    for_runs(true, [&](int64_t a, int64_t b) {
        if ((double)(b - a) * fs + 1e-9 < o.min_speech_sec) std::fill(sp.begin() + a, sp.begin() + b, 0);
    });
    // Prefix sums of speech frames to test a range for speech in O(1).
    std::vector<int64_t> cum((size_t)n + 1, 0);
    for (int64_t f = 0; f < n; ++f) cum[(size_t)f + 1] = cum[(size_t)f] + sp[(size_t)f];
    auto has_speech = [&](int64_t a, int64_t b) { return b > a && cum[(size_t)b] - cum[(size_t)a] > 0; };
    // Pauses: silent runs of at least pause_f frames, in order.
    std::vector<std::pair<int64_t, int64_t>> pauses;
    for_runs(false, [&](int64_t a, int64_t b) { if (b - a >= pause_f) pauses.emplace_back(a, b); });

    // Step 3: cut from the front. A segment starting at s may end at a pause.
    auto emit = [&](int64_t a, int64_t b, double end_sec) {
        if (has_speech(a, std::min(b, n))) out.push_back({(double)a * fs, end_sec});
    };
    int64_t s = 0;
    while (total_sec - (double)s * fs > o.max_seg_sec + 1e-9) {
        const int64_t lo = s + min_f, hi = s + max_f;
        int64_t c = -1, c_mid = -1;
        for (const auto& pz : pauses) {  // ordered, so later pauses overwrite
            const int64_t mid = (pz.first + pz.second) / 2;
            if (pz.first >= lo && pz.second <= hi) c = mid;
            if (mid >= lo && mid <= hi) c_mid = mid;
        }
        if (c < 0) c = c_mid;
        if (c <= s) c = hi;  // hard cut
        emit(s, c, (double)c * fs);
        s = c;
    }
    emit(s, n, total_sec);
    return out;
}

}  // namespace pk
