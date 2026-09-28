#include "scene_render.hpp"
#include "scene_stream.hpp"

#include <cstdio>

using namespace pk;
static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

int main() {
    auto label = [](int i) -> const char* { return i == 0 ? "Speech" : i == 359 ? "Knock" : "Other"; };
    CHECK(format_span(9.0, 10.04) == "[00:09.0 - 00:10.0]");
    CHECK(format_span(75.25, 80.0) == "[01:15.2 - 01:20.0]");
    CHECK(is_speech_label("Speech") && is_speech_label("Male speech, man speaking") && !is_speech_label("Knock"));

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
    if (failures) return 1;
    std::fprintf(stderr, "PASS\n");
    return 0;
}
