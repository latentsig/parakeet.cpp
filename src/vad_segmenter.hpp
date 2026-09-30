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
// pauses found in the per-frame speech probabilities p. See the plan for the
// rule. Every internal boundary is a whole number of frames.
std::vector<VadSegment> segment_by_vad(const std::vector<float>& p, double total_sec,
                                       const SegmenterOpts& o);

}  // namespace pk
