// Unit test for pk::SpeakerIdentifier with a fake embedder. No model or audio.
//
// Fake audio: speaker k is a constant sample value 0.1*(k+1). The fake embedder
// maps the mean sample value back to a one-hot vector, so a clip that mixes two
// speakers (mean 0.15) lands between two voices and would be caught by the checks.
#include "speaker_identifier.hpp"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace pk;

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__);  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static const int kSr = 16000;

struct Fake {
    int calls = 0;
    int last_n = 0;
    std::vector<int> ns;   // every n the embedder was called with, in order
    SpeakerEmbed fn() {
        return [this](const float* pcm, int n, std::vector<float>& emb) {
            ++calls;
            last_n = n;
            ns.push_back(n);
            double sum = 0.0;
            for (int i = 0; i < n; ++i) sum += pcm[i];
            const double mean = n ? sum / n : 0.0;
            const int hot = (int)std::lround(mean * 10.0) - 1;   // 0.1 -> 0, 0.2 -> 1, 0.3 -> 2
            emb.assign(4, 0.0f);
            if (hot >= 0 && hot < 4 && std::fabs(mean * 10.0 - std::round(mean * 10.0)) < 0.2)
                emb[(size_t)hot] = 1.0f;
            else
                emb[3] = 1.0f;   // ambiguous audio (a mixture) goes to a voice nobody enrolled
            return true;
        };
    }
};

static SpeakerRegistry make_registry() {
    SpeakerRegistry r;
    r.enroll("alice", {1, 0, 0, 0});
    r.enroll("bob", {0, 1, 0, 0});
    return r;
}

// Builds a PCM stream of `total_sec` where each listed segment adds its speaker's value.
struct Seg { int spk; float start, end; };
static std::vector<float> make_pcm(double total_sec, const std::vector<Seg>& segs) {
    std::vector<float> pcm((size_t)(total_sec * kSr), 0.0f);
    for (const Seg& s : segs)
        for (int i = (int)(s.start * kSr); i < (int)(s.end * kSr) && i < (int)pcm.size(); ++i)
            pcm[(size_t)i] += 0.1f * (float)(s.spk + 1);
    return pcm;
}

static SpeakerIdOpts opts() {
    SpeakerIdOpts o;
    o.min_voice_sec = 2.0f;
    o.refresh_sec = 3.0f;
    o.max_voice_sec = 10.0f;
    return o;
}

static void test_clean_intervals() {
    auto len = [](const std::vector<Interval>& v) { double t = 0; for (auto& i : v) t += i.end - i.start; return t; };
    auto a = clean_intervals({0, 4}, {{3, 6}}, 0.2);
    CHECK(a.size() == 1 && std::fabs(a[0].start) < 1e-9 && std::fabs(a[0].end - 3) < 1e-9);
    auto b = clean_intervals({0, 4}, {{1, 2}}, 0.2);
    CHECK(b.size() == 2 && std::fabs(len(b) - 3) < 1e-9);
    CHECK(clean_intervals({0, 4}, {{0, 4}}, 0.2).empty());
    CHECK(clean_intervals({0, 4}, {{0.1, 4}}, 0.2).empty());     // 0.1 s sliver dropped
    auto c = clean_intervals({0, 4}, {{2, 5}, {1, 3}}, 0.2);     // overlapping others merge
    CHECK(c.size() == 1 && std::fabs(c[0].end - 1) < 1e-9);
    auto d = clean_intervals({0, 4}, {}, 0.2);
    CHECK(d.size() == 1 && std::fabs(d[0].end - 4) < 1e-9);
}

static void test_names_two_speakers() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdentifier id(f.fn(), &reg, opts());
    const auto pcm = make_pcm(12, {{0, 0, 5}, {1, 6, 11}});
    id.push_pcm(pcm.data(), 5 * kSr);
    id.update({{0, 0.0f, 5.0f}}, {}, false);
    CHECK(id.name(0).name == "alice");
    CHECK(id.name(1).name.empty());
    id.push_pcm(pcm.data() + 5 * kSr, 6 * kSr);
    id.update({{1, 6.0f, 11.0f}}, {}, false);
    CHECK(id.name(1).name == "bob");
    CHECK(id.names().size() == 2);
    CHECK(id.name(7).name.empty());   // a slot never seen is unknown, not an error
}

static void test_min_voice() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdentifier id(f.fn(), &reg, opts());
    const auto pcm = make_pcm(3, {{0, 0, 1.5}});
    id.push_pcm(pcm.data(), 3 * kSr);
    id.update({{0, 0.0f, 1.5f}}, {}, true);   // 1.5 s < min_voice_sec, even at end of stream
    CHECK(f.calls == 0);
    CHECK(id.name(0).name.empty());
}

static void test_refresh_and_last() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdentifier id(f.fn(), &reg, opts());
    const auto pcm = make_pcm(12, {{0, 0, 2.5}, {0, 3, 4}, {0, 5, 6.5}, {0, 7, 8}, {0, 9, 10}});
    auto feed_to = [&](int from, int to) { id.push_pcm(pcm.data() + from * kSr, (to - from) * kSr); };
    feed_to(0, 3);   id.update({{0, 0.0f, 2.5f}}, {}, false);
    CHECK(f.calls == 1 && id.name(0).name == "alice");           // first time past min_voice
    feed_to(3, 5);   id.update({{0, 3.0f, 4.0f}}, {}, false);
    CHECK(f.calls == 1);                                         // gained 1.0 s < refresh 3 s
    feed_to(5, 7);   id.update({{0, 5.0f, 6.5f}}, {}, false);
    CHECK(f.calls == 1);                                         // gained 2.5 s
    feed_to(7, 9);   id.update({{0, 7.0f, 8.0f}}, {}, false);
    CHECK(f.calls == 2);                                         // gained 3.5 s >= 3 s
    feed_to(9, 11);  id.update({{0, 9.0f, 10.0f}}, {}, true);
    CHECK(f.calls == 3);                                         // end of stream flushes the 1.0 s gained
}

static void test_overlap_skipped() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdentifier id(f.fn(), &reg, opts());
    // Speaker 0 talks 0-4 s, speaker 1 talks 3-6 s (still open when 0 closes).
    const auto pcm = make_pcm(6, {{0, 0, 4}, {1, 3, 6}});
    id.push_pcm(pcm.data(), 4 * kSr);
    id.update({{0, 0.0f, 4.0f}}, {{1, 3.0f, 4.0f}}, false);
    CHECK(f.calls == 1);
    CHECK(std::abs(f.last_n - 3 * kSr) <= 2);   // only the 3 s that speaker 0 had alone
    CHECK(id.name(0).name == "alice");           // mixing 3-4 s would have given mean 0.3 -> unknown
}

static void test_overlap_same_call_close() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdentifier id(f.fn(), &reg, opts());
    const auto pcm = make_pcm(6, {{0, 0, 4}, {1, 3, 6}});
    id.push_pcm(pcm.data(), 6 * kSr);
    id.update({{1, 3.0f, 6.0f}, {0, 0.0f, 4.0f}}, {}, false);   // both close in one call
    CHECK(f.ns.size() == 2);
    if (f.ns.size() == 2) {
        CHECK(std::abs(f.ns[0] - 3 * kSr) <= 2);   // slot 0 (map order): 0-3 s alone
        CHECK(std::abs(f.ns[1] - 2 * kSr) <= 2);   // slot 1: 4-6 s alone
    }
    CHECK(id.name(0).name == "alice");
    CHECK(id.name(1).name == "bob");
}

static void test_overlap_earlier_call_close() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdentifier id(f.fn(), &reg, opts());
    const auto pcm = make_pcm(6, {{0, 0, 4}, {1, 3, 6}});
    id.push_pcm(pcm.data(), 6 * kSr);
    // Slot 0 still open: slot 1 keeps 4-6 s, and slot 0's open 0-4 s is taken
    // now (open segments are consumed), minus the 3-4 s that slot 1 shares.
    id.update({{1, 3.0f, 6.0f}}, {{0, 0.0f, 4.0f}}, false);
    CHECK(id.name(1).name == "bob");
    CHECK(f.ns.size() == 2);
    if (f.ns.size() == 2) {
        CHECK(std::abs(f.ns[0] - 3 * kSr) <= 2);   // slot 0 (map order): 3-4 s overlap excluded
        CHECK(std::abs(f.ns[1] - 2 * kSr) <= 2);   // slot 1: 4-6 s alone
    }
    CHECK(id.name(0).name == "alice");
    id.update({{0, 0.0f, 4.0f}}, {}, false);   // closing adds nothing new: 0-4 s was consumed
    CHECK(f.ns.size() == 2);
    CHECK(id.name(0).name == "alice");
}

static void test_unknown_voice() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdentifier id(f.fn(), &reg, opts());
    const auto pcm = make_pcm(5, {{2, 0, 4}});   // a third voice nobody enrolled
    id.push_pcm(pcm.data(), 5 * kSr);
    id.update({{2, 0.0f, 4.0f}}, {}, true);
    CHECK(f.calls >= 1);
    CHECK(id.name(2).name.empty());
}

static void test_hysteresis() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdOpts o = opts();
    o.max_voice_sec = 3.0f;   // the buffer holds only the newest 3 s, so each refresh sees one voice
    SpeakerIdentifier id(f.fn(), &reg, o);
    // Slot 0 is alice's audio first, then bob's audio arrives on the same slot.
    const auto pcm = make_pcm(30, {{0, 0, 3}, {1, 3, 6}, {1, 6, 9}, {1, 9, 12}});
    auto feed_to = [&](int from, int to) { id.push_pcm(pcm.data() + from * kSr, (to - from) * kSr); };
    feed_to(0, 3);   id.update({{0, 0.0f, 3.0f}}, {}, false);
    CHECK(id.name(0).name == "alice");
    feed_to(3, 6);   id.update({{0, 3.0f, 6.0f}}, {}, false);
    CHECK(id.name(0).name == "alice");                           // bob won once: only pending
    feed_to(6, 9);   id.update({{0, 6.0f, 9.0f}}, {}, false);
    CHECK(id.name(0).name == "bob");                             // bob won twice in a row
}

static void test_hysteresis_reset_by_unknown() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdOpts o = opts();
    o.max_voice_sec = 3.0f;
    SpeakerIdentifier id(f.fn(), &reg, o);
    const auto pcm = make_pcm(30, {{0, 0, 3}, {1, 3, 6}, {2, 6, 9}, {1, 9, 12}});
    auto feed_to = [&](int from, int to) { id.push_pcm(pcm.data() + from * kSr, (to - from) * kSr); };
    feed_to(0, 3);   id.update({{0, 0.0f, 3.0f}}, {}, false);
    feed_to(3, 6);   id.update({{0, 3.0f, 6.0f}}, {}, false);    // bob pending
    feed_to(6, 9);   id.update({{0, 6.0f, 9.0f}}, {}, false);    // unknown voice: breaks the run
    feed_to(9, 12);  id.update({{0, 9.0f, 12.0f}}, {}, false);   // bob again, but only once in a row
    CHECK(id.name(0).name == "alice");
}

static void test_ring_drops_old_audio() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdOpts o = opts();
    o.ring_sec = 5.0f;
    o.max_voice_sec = 4.0f;   // ring_sec must be >= max_voice_sec or the constructor rejects the options
    SpeakerIdentifier id(f.fn(), &reg, o);
    const auto pcm = make_pcm(20, {{0, 0, 3}});
    id.push_pcm(pcm.data(), 20 * kSr);
    id.update({{0, 0.0f, 3.0f}}, {}, false);   // its audio is 17 s old and gone from the ring
    CHECK(f.calls == 0);                        // no audio, no embedding, no crash
    CHECK(id.name(0).name.empty());
}

static void test_embed_failure_throws() {
    const SpeakerRegistry reg = make_registry();
    SpeakerIdentifier id([](const float*, int, std::vector<float>&) { return false; }, &reg, opts());
    const auto pcm = make_pcm(5, {{0, 0, 4}});
    id.push_pcm(pcm.data(), 5 * kSr);
    bool threw = false;
    try { id.update({{0, 0.0f, 4.0f}}, {}, false); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw);
}

static void test_offline() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    const auto pcm = make_pcm(24, {{0, 0, 5}, {1, 6, 11}, {0, 12, 17}, {1, 18, 23}});
    const std::vector<SpeakerSegment> segs = {{0, 0, 5}, {1, 6, 11}, {0, 12, 17}, {1, 18, 23}};
    const auto names = identify_offline(pcm, segs, f.fn(), reg, opts());
    CHECK(names.size() == 2);
    CHECK(names.at(0).name == "alice");
    CHECK(names.at(1).name == "bob");
}


// F1: a slot talking without a pause is named while its segment is still open.
static void test_open_segment_named_before_close() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdentifier id(f.fn(), &reg, opts());
    const auto pcm = make_pcm(8, {{0, 0, 8}});
    for (int t = 1; t <= 5; ++t) {
        id.push_pcm(pcm.data() + (t - 1) * kSr, kSr);
        id.update({}, {{0, 0.0f, (float)t}}, false);
        if (t == 1) CHECK(f.calls == 0 && id.name(0).name.empty());   // 1 s < min_voice
        if (t == 2) CHECK(f.calls == 1 && id.name(0).name == "alice"); // named while still open
        if (t == 3 || t == 4) CHECK(f.calls == 1);                    // gained < refresh
    }
    CHECK(id.names().size() == 1);
    CHECK(f.ns.size() == 2);
    if (f.ns.size() == 2) {
        CHECK(std::abs(f.ns[0] - 2 * kSr) <= 2);   // first embedding at 2.0 s consumed
        CHECK(std::abs(f.ns[1] - 5 * kSr) <= 2);   // refresh after 3 s more
    }
    // The segment closes at 6 s: only 5-6 s is new, 0-5 s was consumed while open.
    id.push_pcm(pcm.data() + 5 * kSr, kSr);
    id.update({{0, 0.0f, 6.0f}}, {}, true);
    CHECK(f.ns.size() == 3);
    if (f.ns.size() == 3) CHECK(std::abs(f.ns[2] - 6 * kSr) <= 2);   // 6 s, not 11 s
    CHECK(id.name(0).name == "alice");
}

// A slot is known (listed by names()) as soon as it has an open segment.
static void test_open_slot_is_known() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdentifier id(f.fn(), &reg, opts());
    const auto pcm = make_pcm(2, {{1, 0, 2}});
    id.push_pcm(pcm.data(), kSr);
    id.update({}, {{1, 0.0f, 1.0f}}, false);
    CHECK(id.names().size() == 1 && id.names().count(1) == 1);
    CHECK(id.name(1).name.empty());
}

// Overlap with another slot that opened while this one was open is masked.
static void test_open_segment_overlap_masked() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdentifier id(f.fn(), &reg, opts());
    // Slot 0 talks 0-6 s, slot 1 talks over it 3-4 s.
    const auto pcm = make_pcm(6, {{0, 0, 6}, {1, 3, 4}});
    id.push_pcm(pcm.data(), 3 * kSr);
    id.update({}, {{0, 0.0f, 3.0f}}, false);
    CHECK(f.ns.size() == 1);
    if (!f.ns.empty()) CHECK(std::abs(f.ns[0] - 3 * kSr) <= 2);
    id.push_pcm(pcm.data() + 3 * kSr, kSr);
    id.update({}, {{0, 0.0f, 4.0f}, {1, 3.0f, 4.0f}}, false);   // 3-4 s is shared: nothing added
    id.push_pcm(pcm.data() + 4 * kSr, kSr);
    id.update({{1, 3.0f, 4.0f}}, {{0, 0.0f, 5.0f}}, false);     // 4-5 s alone
    id.push_pcm(pcm.data() + 5 * kSr, kSr);
    id.update({{0, 0.0f, 6.0f}}, {}, true);                      // 5-6 s alone, end flush
    CHECK(f.ns.size() == 2);
    if (f.ns.size() == 2) CHECK(std::abs(f.ns[1] - 5 * kSr) <= 2);   // 0-3 + 4-6, 3-4 excluded
    CHECK(id.name(0).name == "alice");   // mixing 3-4 s in would give mean 0.133, an unknown voice
    CHECK(id.name(1).name.empty());      // slot 1 never had clean audio
}

// A short clean tail at the growing end of an open segment is not lost: it is
// taken again with the audio that follows it.
static void test_open_short_tail_not_lost() {
    Fake f;
    const SpeakerRegistry reg = make_registry();
    SpeakerIdentifier id(f.fn(), &reg, opts());
    const auto pcm = make_pcm(3, {{0, 0, 3}});
    // 0.1 s per update: every step alone is shorter than the 0.2 s minimum piece.
    for (int k = 1; k <= 25; ++k) {
        id.push_pcm(pcm.data() + (k - 1) * (kSr / 10), kSr / 10);
        id.update({}, {{0, 0.0f, 0.1f * (float)k}}, false);
    }
    CHECK(f.calls == 1);
    if (!f.ns.empty()) CHECK(f.ns[0] >= 2 * kSr);
    CHECK(id.name(0).name == "alice");
}

static void test_validate_opts() {
    CHECK(validate_speaker_opts(SpeakerIdOpts{}).empty());
    SpeakerIdOpts o;
    o.min_voice_sec = 0.0f;   CHECK(!validate_speaker_opts(o).empty());
    o = SpeakerIdOpts{}; o.refresh_sec = -1.0f;   CHECK(!validate_speaker_opts(o).empty());
    o = SpeakerIdOpts{}; o.max_voice_sec = 1.0f;  CHECK(!validate_speaker_opts(o).empty());   // < min_voice
    o = SpeakerIdOpts{}; o.accept_threshold = 1.5f; CHECK(!validate_speaker_opts(o).empty());
    o = SpeakerIdOpts{}; o.margin = -0.1f;        CHECK(!validate_speaker_opts(o).empty());
    o = SpeakerIdOpts{}; o.ring_sec = 5.0f;       CHECK(!validate_speaker_opts(o).empty());   // < max_voice
}

int main() {
    test_clean_intervals();
    test_names_two_speakers();
    test_min_voice();
    test_refresh_and_last();
    test_overlap_skipped();
    test_overlap_same_call_close();
    test_overlap_earlier_call_close();
    test_unknown_voice();
    test_hysteresis();
    test_hysteresis_reset_by_unknown();
    test_ring_drops_old_audio();
    test_embed_failure_throws();
    test_offline();
    test_validate_opts();
    test_open_segment_named_before_close();
    test_open_slot_is_known();
    test_open_segment_overlap_masked();
    test_open_short_tail_not_lost();
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_speaker_identifier: PASS\n");
    return 0;
}
