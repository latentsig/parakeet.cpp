#pragma once
// Standalone voice-activity detection: speech segments (and optionally the
// per-frame probabilities) from the model's own VAD head, as one JSON document.
// Shared by the C-API (parakeet_capi_vad_*) and `parakeet-cli vad`.
#include <string>
#include <vector>

#include "vad_segmenter.hpp"

namespace pk {

class Model;

struct VadRequest {
    enum class Mode { kSpeech, kSegments };
    SegmenterOpts opts;               // threshold, min_pause, min_speech, max_segment
    Mode mode = Mode::kSpeech;
    bool probabilities = false;       // add "probabilities" to the document
};

// Parses an options document: a flat JSON object, or NULL / "" for the
// defaults. Keys: "threshold" (0 < x <= 1), "min_pause", "min_speech",
// "max_segment" (seconds, > 0), "mode" ("speech" or "segments"),
// "probabilities" (bool). Unknown keys and bad values are errors. Returns false
// and sets `err` on failure.
bool parse_vad_options(const char* json, VadRequest& req, std::string& err);

// Runs the VAD head over 16 kHz mono PCM and returns the JSON document
// {"mode","duration","frame_sec","backend","segments":[{"start","end"}],
//  "probabilities":[...]?}. Throws std::runtime_error("model has no VAD head")
// for a model without the head.
std::string vad_to_json(const Model& m, const std::vector<float>& pcm16k, const VadRequest& req);

}  // namespace pk
