#include "diarization.hpp"

#include "audio_io.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"

#include <algorithm>
#include <cmath>
#include <new>
#include <stdexcept>

namespace pk {

std::unique_ptr<DiarizationModel> DiarizationModel::load(const std::string& path) {
    // unique_ptr<DiarizationModel> via private ctor: construct then load.
    std::unique_ptr<DiarizationModel> m(new (std::nothrow) DiarizationModel());
    if (!m) return nullptr;
    if (!m->loader_.load(path)) return nullptr;

    const auto& cfg = m->loader_.config();
    if (cfg.arch != "diarization") {
        // Not a diarization model — caller should use Model for ASR.
        return nullptr;
    }
    if (!cfg.diarization.present) {
        return nullptr;
    }

    // Give the weights a backend buffer ONCE so graphs reference them
    // directly as leaves (zero per-call copy), same as Model::load.
    ensure_weights_realized(m->loader_);

    // Construct the component objects (lightweight views over the ModelLoader).
    m->mel_     = std::make_unique<MelFrontend>(m->loader_);
    m->encoder_ = std::make_unique<DiarizationEncoder>(m->loader_);
    m->head_    = std::make_unique<DiarizationHead>(m->loader_);

    return m;
}

DiarizationResult DiarizationModel::diarize_path(const std::string& wav_path) {
    Audio audio;
    if (!load_audio_16k_mono(wav_path, audio)) {
        throw std::runtime_error("parakeet: failed to load audio: " + wav_path);
    }
    // load_audio_16k_mono already resamples to 16 kHz mono.
    return run(audio.samples);
}

DiarizationResult DiarizationModel::diarize_pcm(
        const std::vector<float>& samples, int sample_rate) {
    if (sample_rate <= 0) {
        throw std::runtime_error("parakeet: invalid sample_rate");
    }
    if (sample_rate == 16000) {
        return run(samples);
    }
    std::vector<float> pcm16k = resample_linear(samples, sample_rate, 16000);
    return run(pcm16k);
}

DiarizationResult DiarizationModel::run(const std::vector<float>& samples) {
    const ParakeetConfig& cfg = loader_.config();

    // 1. Log-mel front end → feats [n_mels, T]
    std::vector<float> feats;
    int n_mels = 0, T = 0;
    mel_->compute(samples, feats, n_mels, T);

    // 2. Diarization encoder → enc_out [d_model, T_enc] (channels-first)
    std::vector<float> enc_out;
    int d_model = 0, T_enc = 0;
    encoder_->forward(feats, n_mels, T, enc_out, d_model, T_enc);

    // 3. Diarization head → probs [n_spk, T_out] (post-sigmoid)
    std::vector<float> probs;
    int n_spk = 0, T_out = 0;
    head_->forward(enc_out, d_model, T_enc, probs, n_spk, T_out);

    // 4. Post-process → speaker segments
    std::vector<SpeakerSegment> segs = postprocess(
        probs, n_spk, T_out,
        cfg.diarization.frame_resolution_sec,
        cfg.diarization.onset_threshold,
        cfg.diarization.offset_threshold);

    DiarizationResult result;
    result.segments = std::move(segs);
    result.n_speakers = n_spk;
    return result;
}

std::vector<SpeakerSegment> DiarizationModel::postprocess(
        const std::vector<float>& probs, int n_spk, int T_out,
        float frame_sec, float onset, float offset) const {

    // NeMo predlist_to_timestamps (from nemo.collections.asr.parts.utils.vad_utils):
    //
    // 1. Hysteresis binarization per speaker:
    //    - OFF → ON when prob >= onset
    //    - ON → OFF when prob < offset
    //    When onset == offset (0.5 for Nemotron-3), this is a simple threshold.
    //
    // 2. Extract contiguous ON segments per speaker.
    //
    // 3. min_duration_on / min_duration_off filtering (defaults 0.0 → no-op).
    //
    // 4. merge_overlap_segment: merges same-speaker segments that overlap
    //    (from padded/chunked inference). Offline single-pass produces no
    //    overlaps, so this is a no-op here.
    //
    // 5. Round timestamps to 2 decimal places.

    // probs is row-major [n_spk, T_out]: probs[s * T_out + t]
    std::vector<SpeakerSegment> segments;

    for (int s = 0; s < n_spk; ++s) {
        const float* p = probs.data() + (size_t)s * T_out;

        bool active = false;
        int start_frame = 0;

        for (int t = 0; t < T_out; ++t) {
            const bool on = (p[t] >= onset);
            if (on && !active) {
                // Hysteresis: OFF → ON at onset threshold
                start_frame = t;
                active = true;
            } else if (!on && active) {
                // Hysteresis: ON → OFF when prob drops below offset.
                // With onset == offset, p[t] < offset ⟺ p[t] < onset ⟺ !on.
                float start_sec = start_frame * frame_sec;
                float end_sec   = t * frame_sec;
                segments.push_back({s, start_sec, end_sec});
                active = false;
            }
        }
        // Close any segment still open at the end of the audio.
        if (active) {
            float start_sec = start_frame * frame_sec;
            float end_sec   = T_out * frame_sec;
            segments.push_back({s, start_sec, end_sec});
        }
    }

    // Sort by start time, then by speaker (NeMo returns segments in start order).
    std::sort(segments.begin(), segments.end(),
              [](const SpeakerSegment& a, const SpeakerSegment& b) {
                  if (a.start != b.start) return a.start < b.start;
                  return a.speaker < b.speaker;
              });

    // Round to 2 decimal places (NeMo uses round(ts, 2)).
    for (auto& seg : segments) {
        seg.start = std::round(seg.start * 100.0f) / 100.0f;
        seg.end   = std::round(seg.end   * 100.0f) / 100.0f;
    }

    return segments;
}

} // namespace pk
