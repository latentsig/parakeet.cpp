#pragma once
#include "model_loader.hpp"
#include "diarization_head.hpp"
#include "diarization_encoder.hpp"
#include "mel.hpp"
#include <memory>
#include <string>
#include <vector>

namespace pk {

// A speaker segment: which speaker was active, and when.
// Timestamps are in seconds, matching ASR Word timestamps for Phase 3 (SAS).
struct SpeakerSegment {
    int speaker;    // 0-indexed speaker ID (0..n_speakers-1)
    float start;    // segment start in seconds
    float end;      // segment end in seconds
};

// Diarization result: list of speaker segments + metadata.
struct DiarizationResult {
    std::vector<SpeakerSegment> segments;
    int n_speakers;  // max speakers the model supports
};

// DiarizationModel — offline speaker diarization ("who spoke when").
//
// Composes a DiarizationEncoder (pre-LN RoPE Transformer for Nemotron-3-
// Diarization) with a DiarizationHead (sortformer speaker sigmoid head).
// The mel frontend is shared with the ASR path.
//
// The model is loaded from a GGUF with arch="diarization". The C++ loader
// reuses ModelLoader (shared with ASR) and constructs a DiarizationEncoder +
// MelFrontend + DiarizationHead — no CTC/RNNT decoder.
class DiarizationModel {
public:
    // Load a diarization GGUF. Returns nullptr on failure.
    static std::unique_ptr<DiarizationModel> load(const std::string& path);

    // Diarize an audio file (any format audio_io supports). Returns segments.
    DiarizationResult diarize_path(const std::string& wav_path);

    // Diarize raw PCM samples (mono float, any sample rate — resampled to 16k).
    DiarizationResult diarize_pcm(const std::vector<float>& samples,
                                   int sample_rate);

    const ParakeetConfig& config() const { return loader_.config(); }
    const ModelLoader& loader() const { return loader_; }

private:
    DiarizationModel() = default;

    // Internal: run the full pipeline (mel → encoder → head → postprocess)
    // on already-16kHz PCM.
    DiarizationResult run(const std::vector<float>& samples);

    // Post-process per-frame speaker probabilities into speaker segments.
    // Matches NeMo's predlist_to_timestamps: hysteresis binarization
    // (onset/offset thresholds), segment extraction, min_duration filtering,
    // merge overlapping segments, round to 2 decimal places.
    std::vector<SpeakerSegment> postprocess(const std::vector<float>& probs,
                                             int n_spk, int T_out,
                                             float frame_sec,
                                             float onset, float offset) const;

    ModelLoader loader_;
    std::unique_ptr<MelFrontend> mel_;
    std::unique_ptr<DiarizationEncoder> encoder_;
    std::unique_ptr<DiarizationHead> head_;
};

} // namespace pk
