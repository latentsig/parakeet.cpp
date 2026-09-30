#pragma once
#include "diarization.hpp"        // pk::SpeakerSegment
#include "speaker_registry.hpp"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace pk {

struct SpeakerIdOpts {
    float min_voice_sec = 2.0f;       // clean audio a slot needs before it is embedded
    float refresh_sec = 3.0f;         // new clean audio that triggers another embedding
    float max_voice_sec = 10.0f;      // a slot's newest audio kept for embedding
    float accept_threshold = 0.5f;    // minimum cosine to take a name
    float margin = 0.05f;             // best must beat the runner-up by this much
    float ring_sec = 60.0f;           // PCM history kept to slice segments out of
};

// "" when valid, else what is wrong. All of: min_voice_sec > 0, refresh_sec > 0,
// max_voice_sec >= min_voice_sec, ring_sec >= max_voice_sec, accept_threshold in
// [-1, 1], margin >= 0.
std::string validate_speaker_opts(const SpeakerIdOpts& o);

// A diarization slot's current identity. Empty name = unknown.
struct SlotName {
    std::string name;
    float score = 0.0f;
};

// Embeds one 16 kHz mono window. False on failure.
using SpeakerEmbed = std::function<bool(const float* pcm, int n, std::vector<float>& emb)>;

struct Interval {
    double start;
    double end;
};

// The parts of `seg` not covered by any of `others`, each at least `min_len` long.
std::vector<Interval> clean_intervals(const Interval& seg, const std::vector<Interval>& others,
                                      double min_len);

// Names diarization slots by embedding each slot's clean (single speaker) audio
// and matching it against a registry. Drive it with the stream's PCM and the
// diarizer's segments; it never talks to a model itself. Not thread-safe.
class SpeakerIdentifier {
public:
    // `registry` is borrowed and must outlive the identifier.
    SpeakerIdentifier(SpeakerEmbed embed, const SpeakerRegistry* registry, SpeakerIdOpts opts);

    // Appends 16 kHz mono PCM (the same audio diarization sees).
    void push_pcm(const float* pcm, int n);
    // `closed`: segments that closed since the last call. `open`: segments still
    // open now (used only to skip overlap). Throws std::runtime_error when the
    // embed callback fails.
    // Contract: `open` must list every segment that has started and not yet
    // closed, with `end` at least the end of any segment closing in this call
    // that it overlaps. Overlap with a segment that was neither in `open` nor
    // already closed is embedded and never revisited.
    void update(const std::vector<SpeakerSegment>& closed, const std::vector<SpeakerSegment>& open,
                bool is_last);

    SlotName name(int slot) const;               // unknown for a slot never seen
    std::map<int, SlotName> names() const;       // every slot that has been seen

private:
    struct Slot {
        std::vector<float> voice;   // newest clean audio, at most max_voice_sec
        double gained_sec = 0.0;    // clean audio added since the last embedding
        bool embedded = false;
        SlotName current;
        std::string pending;        // a different known name that won last time
    };

    void add_audio(int slot, const Interval& iv);
    void maybe_embed(Slot& s, bool is_last);
    void apply(Slot& s, const SpeakerMatch& m);

    SpeakerEmbed embed_;
    const SpeakerRegistry* registry_;
    SpeakerIdOpts opts_;
    std::vector<float> ring_;
    long long ring_base_ = 0;        // absolute sample index of ring_[0]
    long long total_ = 0;            // samples pushed so far
    std::vector<SpeakerSegment> history_;   // recent closed segments, for overlap checks
    std::map<int, Slot> slots_;
};

// Names slots for a finished recording: runs the same logic once over all
// segments. Returns every slot that appears in `segs`. It raises max_voice_sec
// to at least 30 s and ring_sec to cover the whole recording.
std::map<int, SlotName> identify_offline(const std::vector<float>& pcm16k,
                                         const std::vector<SpeakerSegment>& segs,
                                         const SpeakerEmbed& embed, const SpeakerRegistry& reg,
                                         const SpeakerIdOpts& opts);

}  // namespace pk
