#include "vad_segmenter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace pk {

std::vector<VadSegment> segment_by_vad(const std::vector<float>& p, double total_sec,
                                       const SegmenterOpts& o) {
    std::vector<VadSegment> out;
    // Degenerate options: no usable frame grid, so do not cut at all.
    if (!(o.frame_sec > 0.0) || !std::isfinite(o.frame_sec) || !std::isfinite(o.max_seg_sec) ||
        !(o.max_seg_sec > 2.0 * o.frame_sec) || !std::isfinite(total_sec)) {
        out.push_back({0.0, total_sec});
        return out;
    }
    if (total_sec <= o.max_seg_sec) {
        out.push_back({0.0, total_sec});
        return out;
    }
    const double fs = o.frame_sec;
    const int64_t n = (int64_t)p.size();
    const int64_t max_f = std::max<int64_t>(2, (int64_t)std::floor(o.max_seg_sec / fs + 1e-9));
    const int64_t min_f = std::min<int64_t>(max_f - 1, std::max<int64_t>(1, (int64_t)std::ceil(o.min_seg_sec / fs - 1e-9)));
    const int64_t pause_f = std::max<int64_t>(1, (int64_t)std::ceil(o.min_pause_sec / fs - 1e-9));
    auto silent = [&](int64_t f) { return f >= n || p[(size_t)f] < o.threshold; };

    // Middle of the longest silent run inside [lo, hi) that is at least pause_f
    // frames long (ties go to the later run); -1 if there is none.
    auto best_cut = [&](int64_t lo, int64_t hi) -> int64_t {
        int64_t best = -1, best_len = 0;
        int64_t f = lo;
        while (f < hi) {
            if (!silent(f)) { ++f; continue; }
            int64_t e = f;
            while (e < hi && silent(e)) ++e;
            const int64_t len = e - f;
            if (len >= pause_f && len >= best_len) { best_len = len; best = (f + e) / 2; }
            f = e;
        }
        return best;
    };

    int64_t s = 0;
    while (total_sec - (double)s * fs > o.max_seg_sec + 1e-9) {
        const int64_t hi = s + max_f;
        int64_t c = best_cut(s + (max_f * 2) / 3, hi);
        if (c < 0) c = best_cut(s + min_f, hi);
        if (c <= s) c = hi;  // hard cut
        out.push_back({(double)s * fs, (double)c * fs});
        s = c;
    }
    out.push_back({(double)s * fs, total_sec});
    return out;
}

}  // namespace pk
