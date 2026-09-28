// Unit test for the SAS merge layer (merge_asr_diarization + group_speaker_words).
//
// No model or audio needed — constructs synthetic Word and SpeakerSegment vectors
// and verifies the merge + grouping logic directly.

#include "sas_merge.hpp"
#include "transcription.hpp"
#include "diarization.hpp"

#include <cassert>
#include <cstdio>

using namespace pk;

static int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

// ── Test 1: basic overlap assignment ──────────────────────────────────────
// Two speakers, non-overlapping segments. Words fall clearly within each
// speaker's segment.
static void test_basic_assignment() {
    std::vector<Word> words = {
        {"hello",   0.10f, 0.30f, 0.95f},
        {"world",   0.35f, 0.55f, 0.90f},
        {"foo",     1.10f, 1.30f, 0.80f},
        {"bar",     1.35f, 1.55f, 0.85f},
    };
    std::vector<SpeakerSegment> segs = {
        {0, 0.00f, 1.00f},  // speaker 0: 0–1s
        {1, 1.00f, 2.00f},  // speaker 1: 1–2s
    };

    auto swords = merge_asr_diarization(words, segs);
    CHECK(swords.size() == 4);
    CHECK(swords[0].speaker == 0);
    CHECK(swords[1].speaker == 0);
    CHECK(swords[2].speaker == 1);
    CHECK(swords[3].speaker == 1);

    // Verify text and timestamps are preserved
    CHECK(swords[0].text == "hello");
    CHECK(swords[0].start == 0.10f);
    CHECK(swords[0].end == 0.30f);
    CHECK(swords[0].conf == 0.95f);
}

// ── Test 2: dominant speaker (overlap) ────────────────────────────────────
// Two overlapping segments: the one with the larger overlap wins.
static void test_dominant_speaker() {
    std::vector<Word> words = {
        {"x", 0.50f, 0.80f, 0.9f},  // overlaps both, but more with spk 0
    };
    std::vector<SpeakerSegment> segs = {
        {0, 0.00f, 0.70f},  // overlap = 0.20s
        {1, 0.60f, 1.00f},  // overlap = 0.20s — equal! first one (sorted) wins
    };

    auto swords = merge_asr_diarization(words, segs);
    CHECK(swords.size() == 1);
    // Equal overlap → the first segment (sorted by start, then speaker) wins
    // because we use strict > (not >=).
    CHECK(swords[0].speaker == 0);

    // Now make speaker 1 overlap more
    segs[1].start = 0.40f;  // overlap with [0.5, 0.8] = 0.40s > 0.20s
    swords = merge_asr_diarization(words, segs);
    CHECK(swords[0].speaker == 1);
}

// ── Test 3: no overlapping segment ────────────────────────────────────────
// Word falls outside all segments → speaker = -1.
static void test_no_speaker() {
    std::vector<Word> words = {
        {"silence", 5.00f, 5.50f, 0.5f},
    };
    std::vector<SpeakerSegment> segs = {
        {0, 0.00f, 1.00f},
        {1, 2.00f, 3.00f},
    };

    auto swords = merge_asr_diarization(words, segs);
    CHECK(swords.size() == 1);
    CHECK(swords[0].speaker == -1);
}

// ── Test 3b: no overlap, but a segment within the snap distance ─────────
// ASR and diarization boundaries can disagree slightly: a word that just
// misses a segment takes the nearest segment's speaker.
static void test_snap_to_nearest() {
    std::vector<Word> words = {
        {"well", 19.92f, 20.00f, 0.7f},   // 0.10 s before spk 1 starts
        {"far", 25.00f, 25.20f, 0.7f},    // 1.4 s after spk 1 ends
    };
    std::vector<SpeakerSegment> segs = {
        {0, 14.78f, 18.75f},              // 1.17 s away from "well"
        {1, 20.10f, 23.60f},
    };
    auto swords = merge_asr_diarization(words, segs);
    CHECK(swords[0].speaker == 1);
    CHECK(swords[1].speaker == -1);
    // Snapping can be disabled.
    swords = merge_asr_diarization(words, segs, 0.0f);
    CHECK(swords[0].speaker == -1);
}

// ── Test 4: utterance grouping — same speaker, small gap ──────────────────
static void test_grouping_same_speaker() {
    std::vector<SpeakerWord> swords = {
        {0, "hello", 0.10f, 0.30f, 0.95f},
        {0, "world", 0.40f, 0.60f, 0.90f},  // gap = 0.10s ≤ 0.5s
    };

    auto utts = group_speaker_words(swords);
    CHECK(utts.size() == 1);
    CHECK(utts[0].speaker == 0);
    CHECK(utts[0].text == "hello world");
    CHECK(utts[0].start == 0.10f);
    CHECK(utts[0].end == 0.60f);
    CHECK(utts[0].conf == 0.90f);  // min(0.95, 0.90)
}

// ── Test 5: utterance grouping — speaker change ───────────────────────────
static void test_grouping_speaker_change() {
    std::vector<SpeakerWord> swords = {
        {0, "hello", 0.10f, 0.30f, 0.95f},
        {1, "world", 0.40f, 0.60f, 0.90f},
    };

    auto utts = group_speaker_words(swords);
    CHECK(utts.size() == 2);
    CHECK(utts[0].speaker == 0);
    CHECK(utts[0].text == "hello");
    CHECK(utts[1].speaker == 1);
    CHECK(utts[1].text == "world");
}

// ── Test 6: utterance grouping — large gap splits ─────────────────────────
static void test_grouping_large_gap() {
    std::vector<SpeakerWord> swords = {
        {0, "hello", 0.10f, 0.30f, 0.95f},
        {0, "world", 1.00f, 1.20f, 0.90f},  // gap = 0.70s > 0.5s
    };

    auto utts = group_speaker_words(swords);
    CHECK(utts.size() == 2);
    CHECK(utts[0].text == "hello");
    CHECK(utts[1].text == "world");
}

// ── Test 7: utterance grouping — unknown speaker (-1) ─────────────────────
static void test_grouping_unknown_speaker() {
    std::vector<SpeakerWord> swords = {
        {0, "a", 0.10f, 0.20f, 0.9f},
        {-1, "b", 0.30f, 0.40f, 0.8f},
        {-1, "c", 0.45f, 0.55f, 0.7f},
        {1, "d", 0.60f, 0.70f, 0.85f},
    };

    auto utts = group_speaker_words(swords);
    CHECK(utts.size() == 3);
    CHECK(utts[0].speaker == 0);
    CHECK(utts[0].text == "a");
    CHECK(utts[1].speaker == -1);
    CHECK(utts[1].text == "b c");
    CHECK(utts[2].speaker == 1);
    CHECK(utts[2].text == "d");
}

// ── Test 8: empty inputs ──────────────────────────────────────────────────
static void test_empty() {
    std::vector<Word> words;
    std::vector<SpeakerSegment> segs;

    auto swords = merge_asr_diarization(words, segs);
    CHECK(swords.empty());

    auto utts = group_speaker_words(swords);
    CHECK(utts.empty());
}

// ── Test 9: word exactly at segment boundary ───────────────────────────────
static void test_boundary() {
    std::vector<Word> words = {
        {"x", 1.00f, 1.10f, 0.9f},  // starts exactly at seg0 end / seg1 start
    };
    std::vector<SpeakerSegment> segs = {
        {0, 0.00f, 1.00f},  // seg.end <= w.start → skipped (<=)
        {1, 1.00f, 2.00f},  // overlap = 0.10s
    };

    auto swords = merge_asr_diarization(words, segs);
    CHECK(swords.size() == 1);
    CHECK(swords[0].speaker == 1);
}

// ── Test 10: multiple speakers on same segment ────────────────────────────
// Same segment list, multiple words — all should get the same speaker.
static void test_multiple_words_same_speaker() {
    std::vector<Word> words = {
        {"a", 0.10f, 0.20f, 0.9f},
        {"b", 0.25f, 0.35f, 0.8f},
        {"c", 0.40f, 0.50f, 0.7f},
    };
    std::vector<SpeakerSegment> segs = {
        {2, 0.00f, 1.00f},
    };

    auto swords = merge_asr_diarization(words, segs);
    CHECK(swords.size() == 3);
    for (int i = 0; i < 3; ++i) {
        CHECK(swords[i].speaker == 2);
    }

    // Group should merge all into one utterance
    auto utts = group_speaker_words(swords);
    CHECK(utts.size() == 1);
    CHECK(utts[0].text == "a b c");
    CHECK(utts[0].speaker == 2);
    CHECK(utts[0].conf == 0.7f);  // min
}

int main() {
    test_basic_assignment();
    test_dominant_speaker();
    test_no_speaker();
    test_snap_to_nearest();
    test_grouping_same_speaker();
    test_grouping_speaker_change();
    test_grouping_large_gap();
    test_grouping_unknown_speaker();
    test_empty();
    test_boundary();
    test_multiple_words_same_speaker();

    if (failures == 0) {
        std::printf("All SAS merge tests passed.\n");
        return 0;
    }
    std::printf("%d assertion(s) failed.\n", failures);
    return 1;
}
