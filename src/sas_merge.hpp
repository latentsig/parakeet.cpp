#pragma once
#include "transcription.hpp"
#include "diarization.hpp"

#include <string>
#include <vector>

namespace pk {

// A word attributed to a speaker — the output of merging ASR
// transcription with diarization segments.
struct SpeakerWord {
    int         speaker;   // from diarization (0-based, -1 = no speaker)
    std::string text;      // from ASR
    float       start;     // from ASR word (seconds)
    float       end;       // from ASR word (seconds)
    float       conf;      // from ASR word
};

// A speaker-attributed utterance: consecutive words from the same speaker
// that form a phrase. Groups SpeakerWords where the speaker doesn't change
// and the gap between words is small.
struct SpeakerUtterance {
    int         speaker;
    std::string text;      // space-joined words
    float       start;     // first word start
    float       end;       // last word end
    float       conf;      // min word confidence
};

// Merge ASR word timestamps with diarization speaker segments.
//
// For each word, the dominant active speaker is the one whose diarization
// segment overlaps the word's [start, end] interval by the largest amount.
// Words with no overlapping segment get speaker = -1.
//
// `word_frame_sec` and `diar_frame_sec` are the ASR and diarization encoder
// frame strides (seconds per encoder frame). They are not used for the merge
// itself (timestamps are already in seconds) but are exposed in the signature
// for future streaming use where frame-level alignment is needed.
std::vector<SpeakerWord> merge_asr_diarization(
    const std::vector<Word>& words,
    const std::vector<SpeakerSegment>& segs);

// Group speaker-attributed words into utterances.
// Consecutive words with the same speaker and gap <= max_gap_sec are joined.
// speaker == -1 words are grouped together as "unknown" utterances.
std::vector<SpeakerUtterance> group_speaker_words(
    const std::vector<SpeakerWord>& swords,
    float max_gap_sec = 0.5f);

} // namespace pk
