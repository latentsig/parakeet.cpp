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
    double min_pause_sec = 0.2;   // shortest silence that may be cut in
    double min_seg_sec = 1.0;     // earliest cut position inside a segment
    double bridge_sec = 0.1;      // speech gaps shorter than this are bridged
    double min_speech_sec = 0.1;  // speech runs shorter than this are dropped
};

// Cuts [0, total_sec] into segments of at most max_seg_sec, at pauses found in
// the per-frame speech probabilities p.
//
// 1. A frame is speech when p >= threshold. Speech gaps shorter than bridge_sec
//    are filled, then speech runs shorter than min_speech_sec are removed. The
//    remaining silent runs of at least min_pause_sec are the pauses.
// 2. Audio of at most max_seg_sec is returned whole, with or without speech.
// 3. Longer audio is cut from the front. For a segment starting at s the cut is
//    the midpoint of the last pause that lies fully inside [s + min_seg_sec,
//    s + max_seg_sec]; else the midpoint of the last pause whose midpoint lies
//    in that range; else a hard cut at s + max_seg_sec.
// 4. Segments that contain no speech are dropped, including the trailing
//    remainder. The result is empty when no segment has speech. Kept segments
//    are ordered and disjoint but need not touch.
//
// Degenerate options (frame_sec not finite or <= 0, max_seg_sec not finite or
// <= 2 * frame_sec, threshold or any of the four durations not finite, any
// duration above 1e6 seconds) return the single segment {0, total_sec}. Every
// internal boundary is a whole number of frames.
std::vector<VadSegment> segment_by_vad(const std::vector<float>& p, double total_sec,
                                       const SegmenterOpts& o);

// The speech regions themselves, for any audio length (no cap, no cuts).
// Uses threshold, frame_sec, bridge_sec, min_speech_sec and min_pause_sec: the
// speech mask is smoothed as in segment_by_vad (bridge gaps shorter than
// bridge_sec, drop runs shorter than min_speech_sec), then speech runs
// separated by a gap shorter than min_pause_sec are merged. Regions are
// ordered, disjoint and inside [0, total_sec]; the result is empty when there
// is no speech. Degenerate options return the single region {0, total_sec}.
std::vector<VadSegment> speech_regions(const std::vector<float>& p, double total_sec,
                                       const SegmenterOpts& o);

}  // namespace pk
