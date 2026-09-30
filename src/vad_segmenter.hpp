#pragma once
#include <vector>

namespace pk {

struct VadSegment {
    double start = 0.0;  // seconds
    double end = 0.0;
};

struct SegmenterOpts {
    float threshold = 0.5f;       // frame is speech when p >= threshold
    double frame_sec = 0.08;
    double max_seg_sec = 30.0;
    double min_pause_sec = 0.32;  // shortest silence that may be cut in
    double min_seg_sec = 8.0;     // do not cut earlier than this into a segment
};

// Cuts [0, total_sec] into contiguous segments of at most max_seg_sec, at
// pauses found in the per-frame speech probabilities p. Rule: cut in the middle
// of the longest run of frames below threshold (at least min_pause_sec long),
// searched first in the last third of the allowed window then in the whole
// window from min_seg_sec on; ties go to the later run; hard cut at max_seg_sec
// if no pause is found. Degenerate options (frame_sec not finite or <= 0,
// max_seg_sec not finite or <= 2 * frame_sec, threshold, min_seg_sec or
// min_pause_sec not finite, any of the three durations above 1e6 seconds) return the single segment
// {0, total_sec}. Every internal boundary is a whole number of frames.
std::vector<VadSegment> segment_by_vad(const std::vector<float>& p, double total_sec,
                                       const SegmenterOpts& o);

}  // namespace pk
