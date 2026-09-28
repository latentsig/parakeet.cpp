#pragma once
#include "asr_committer.hpp"
#include "diar_pcm_stream.hpp"
#include "sas_merge.hpp"      // pk::SpeakerWord, pk::SpeakerUtterance
#include "sound_stream.hpp"   // pk::SoundOpts, pk::SoundSegment, pk::SoundWindow

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace pk {

class Model;
class CedTagger;

// The models a scene stream runs over. Any may be null; at least one is needed.
struct SceneParts {
    const Model* asr = nullptr;
    const DiarizationModel* diar = nullptr;
    DiarLatency diar_latency = DiarLatency::Model;
    CedTagger* tagger = nullptr;     // sound events
    SoundOpts sound;                 // sound events
};

// What one feed finalized. All times are seconds on the stream clock.
struct SceneUpdate {
    double t = 0;                    // stream time consumed
    double safe_until = 0;           // renderer: nothing later can start before this
    std::vector<SpeakerUtterance> utterances;
    std::vector<SpeakerWord> words;
    std::vector<SpeakerSegment> speakers;           // closed this call
    std::vector<SoundSegment> sounds;               // closed this call
    std::vector<StreamingSpeakerSegment> active_speakers;
    std::vector<SoundSegment> active_sounds;
};

// The part running when feed() threw, so a caller can attribute the error.
enum class ScenePart { None, Diarization, Asr, Sound };

// Speech, speakers and sound events over one live 16 kHz mono PCM stream.
// Each feed gives the PCM to every part, then collects what each one
// finalized. Not thread-safe.
class SceneStream {
public:
    explicit SceneStream(const SceneParts& p);      // throws std::invalid_argument when no part is given
    ~SceneStream();
    SceneStream(const SceneStream&) = delete;
    SceneStream& operator=(const SceneStream&) = delete;

    SceneUpdate feed(const float* pcm, int n, bool is_last);
    std::vector<SoundWindow> drain_windows();        // empty without a tagger
    // True after an is_last feed. With diarization it turns true as soon as
    // the diarizer takes the is_last chunk, so an is_last feed that throws
    // later (diarizer or transcriber) still ends the stream and is not re-run.
    bool finished() const { return finished_ || (diar_ && diar_->finished()); }
    const DiarPcmStream* diar() const { return diar_.get(); }
    // The part that was running when the last feed() threw.
    ScenePart failed_part() const { return part_; }

private:
    std::unique_ptr<DiarPcmStream> diar_;
    std::unique_ptr<AsrCommitter> asr_;
    std::unique_ptr<SoundStream> sound_;
    std::vector<SpeakerSegment> segs_;   // closed diarization segments not yet behind the commit point
    double t_ = 0.0;                     // stream time consumed
    bool finished_ = false;
    ScenePart part_ = ScenePart::None;
};

// Serialize a SceneUpdate to the scene stream's JSON document shape:
// {"t","utterances","words","speakers","sounds",
//  "active":{"speakers","sounds"}}. `label(i)` may return nullptr (emitted
// as ""); pass a function that always returns nullptr when there is no
// tagger (sounds are then always empty, so it is never called).
std::string scene_update_to_json(const SceneUpdate& u, const std::function<const char*(int)>& label);

} // namespace pk
