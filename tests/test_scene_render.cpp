#include "scene_render.hpp"
#include "scene_stream.hpp"

#include <cstdio>

using namespace pk;
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

int main() {
    auto label = [](int i) -> const char* {
        return i == 0 ? "Speech" : i == 359 ? "Knock" : i == 42 ? "Speech synthesizer" : "Other";
    };
    CHECK(format_span(9.0, 10.04) == "[00:09.0 - 00:10.0]");
    CHECK(format_span(75.25, 80.0) == "[01:15.2 - 01:20.0]");
    CHECK(is_speech_label("Speech") && is_speech_label("Male speech, man speaking") && !is_speech_label("Knock"));
    CHECK(is_speech_label("Speech synthesizer"));

    SceneRenderer r(/*has_diar=*/true, /*show_speech=*/false, label);
    SceneUpdate u1;
    u1.sounds = {{359, 9.0f, 10.0f, 0.81f}, {0, 10.5f, 11.8f, 0.9f}};
    u1.utterances = {{0, "who is it", 10.9f, 11.6f, 0.93f}};
    r.add(u1);
    auto early = r.flush(10.0);
    CHECK(early.size() == 1 && early[0] == "[00:09.0 - 00:10.0]  (Knock 0.81)");
    auto rest = r.flush_all();
    CHECK(rest.size() == 1 && rest[0] == "[00:10.9 - 00:11.6]  Speaker 0: who is it");  // Speech hidden

    SceneRenderer nd(/*has_diar=*/false, /*show_speech=*/true, label);
    SceneUpdate u2;
    u2.utterances = {{-1, "hello", 1.0f, 1.5f, 0.9f}};
    u2.sounds = {{0, 0.5f, 2.0f, 0.9f}};
    nd.add(u2);
    auto lines = nd.flush_all();
    CHECK(lines.size() == 2 && lines[0] == "[00:00.5 - 00:02.0]  (Speech 0.90)" &&
          lines[1] == "[00:01.0 - 00:01.5]  hello");

    // "Speech synthesizer" is a child of Speech in the AudioSet ontology; CED
    // emits it over clean narration, so it hides the same way "Speech" does.
    SceneRenderer hide_synth(/*has_diar=*/false, /*show_speech=*/false, label);
    SceneUpdate u3;
    u3.sounds = {{42, 2.0f, 4.0f, 0.8f}};
    hide_synth.add(u3);
    CHECK(hide_synth.flush_all().empty());

    SceneRenderer show_synth(/*has_diar=*/false, /*show_speech=*/true, label);
    show_synth.add(u3);
    auto synth_lines = show_synth.flush_all();
    CHECK(synth_lines.size() == 1 && synth_lines[0] == "[00:02.0 - 00:04.0]  (Speech synthesizer 0.80)");

    // Diarization without ASR: closed speaker segments become lines, ordered
    // with sounds, and held while an earlier-starting segment is still open.
    SceneRenderer dz(/*has_diar=*/true, /*show_speech=*/false, label, /*has_asr=*/false);
    SceneUpdate d1;
    d1.speakers = {{1, 2.0f, 4.0f}};
    d1.active_speakers = {{0, 1.0f, 5.0f}};   // speaker 0 open since 1.0 s
    d1.sounds = {{359, 3.0f, 3.5f, 0.7f}};
    dz.add(d1);
    CHECK(dz.flush(5.0).empty());               // speaker 0 may still close with start 1.0
    SceneUpdate d2;
    d2.speakers = {{0, 1.0f, 6.0f}};
    d2.active_speakers = {{1, 6.5f, 7.0f}};
    dz.add(d2);
    auto dl = dz.flush(7.0);
    CHECK(dl.size() == 3 && dl[0] == "[00:01.0 - 00:06.0]  Speaker 0" &&
          dl[1] == "[00:02.0 - 00:04.0]  Speaker 1" && dl[2] == "[00:03.0 - 00:03.5]  (Knock 0.70)");
    SceneUpdate d3;
    d3.speakers = {{1, 6.5f, 8.0f}};
    dz.add(d3);
    auto dr = dz.flush_all();
    CHECK(dr.size() == 1 && dr[0] == "[00:06.5 - 00:08.0]  Speaker 1");

    // With ASR, speaker segments are not printed (utterances carry the speaker).
    SceneRenderer withasr(/*has_diar=*/true, /*show_speech=*/false, label);
    withasr.add(d1);
    auto wl = withasr.flush_all();
    CHECK(wl.size() == 1 && wl[0] == "[00:03.0 - 00:03.5]  (Knock 0.70)");

    if (failures) return 1;
    std::fprintf(stderr, "PASS\n");
    return 0;
}
