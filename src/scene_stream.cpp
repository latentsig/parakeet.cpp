#include "scene_stream.hpp"

#include "model.hpp"  // pk::Model

#include <algorithm>
#include <stdexcept>

namespace pk {

// A scene with only a tagger is accepted but does nothing yet: the sound
// part is not wired into feed().
SceneStream::SceneStream(const SceneParts& p) {
    if (!p.asr && !p.diar && !p.tagger)
        throw std::invalid_argument("scene stream needs at least one model");
    if (p.diar) diar_ = std::make_unique<DiarPcmStream>(*p.diar, p.diar_latency);
    if (p.asr) {
        const Model* m = p.asr;
        asr_ = std::make_unique<AsrCommitter>(
            [m](const std::vector<float>& x) { return m->transcribe_with_timestamps(x, 16000).words; });
    }
}

SceneStream::~SceneStream() = default;

SceneUpdate SceneStream::feed(const float* pcm, int n, bool is_last) {
    SceneUpdate u;
    std::vector<StreamingSpeakerSegment> closed;
    long long advanced = 0;
    // Until the ASR transcribes, a failure is charged to diarization (as the
    // SAS C-API always did), including the ASR audio push.
    part_ = diar_ ? ScenePart::Diarization : ScenePart::Asr;
    if (diar_) {
        advanced = diar_->feed(pcm, n, is_last, closed);
        for (const auto& c : closed) {
            segs_.push_back({c.speaker, c.start, c.end});
            u.speakers.push_back({c.speaker, c.start, c.end});
        }
    }
    if (asr_) {
        asr_->push(pcm, n);
        // With diarization, ASR follows how far diarization has got, and only
        // when it advanced (the SAS behavior). Without it, the audio received.
        // A commit that the min window skips changes nothing, not even the
        // segment pruning below (the SAS behavior).
        const bool tick = !diar_ || advanced > 0 || is_last;
        const double until = diar_ ? diar_->diarized_until() : t_ + (n > 0 ? n : 0) / 16000.0;
        if (tick && asr_->ready(until, is_last)) {
            part_ = ScenePart::Asr;
            auto committed = asr_->commit(until, is_last);
            // Speaker segments known so far: closed ones plus those still open.
            std::vector<SpeakerSegment> segs = segs_;
            if (diar_)
                for (const auto& o : diar_->open_segments()) segs.push_back({o.speaker, o.start, o.end});
            u.words = merge_asr_diarization(committed, segs);
            u.utterances = group_speaker_words(u.words);
            // Segments that ended before the commit point can no longer match a word.
            const double commit_sec = asr_->commit_sec();
            segs_.erase(std::remove_if(segs_.begin(), segs_.end(),
                                       [&](const SpeakerSegment& g) { return g.end < commit_sec; }),
                        segs_.end());
        }
    }
    t_ += (n > 0 ? n : 0) / 16000.0;
    if (is_last) finished_ = true;
    u.t = t_;
    if (diar_) {
        part_ = ScenePart::Diarization;
        u.active_speakers = diar_->open_segments();
    }
    u.safe_until = finished_ ? t_ : (asr_ ? asr_->commit_sec() : t_);
    part_ = ScenePart::None;
    return u;
}

std::vector<SoundWindow> SceneStream::drain_windows() { return {}; }

} // namespace pk
