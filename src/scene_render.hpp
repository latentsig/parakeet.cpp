#pragma once
#include "scene_stream.hpp"  // pk::SceneUpdate

#include <functional>
#include <string>
#include <vector>

namespace pk {

// True for a CED label the scene renderer treats as plain speech (already
// carried by the ASR/diarization transcript, so it is redundant on screen
// unless the caller asks to see it).
bool is_speech_label(const std::string& label);

// "[mm:ss.s - mm:ss.s]", tenths truncated (not rounded).
std::string format_span(double start, double end);

// Turns SceneUpdate pieces into printable transcript lines, time-ordered.
// Not thread-safe; one scene stream at a time.
class SceneRenderer {
public:
    SceneRenderer(bool has_diar, bool show_speech, std::function<const char*(int)> label);

    // Queues the lines for one SceneUpdate's utterances and sounds.
    void add(const SceneUpdate& u);

    // Returns (and removes) queued lines with start < safe_until, time-ordered.
    std::vector<std::string> flush(double safe_until);
    // Returns (and removes) every remaining queued line, time-ordered.
    std::vector<std::string> flush_all();

private:
    struct Item {
        double start;
        std::string line;
    };

    bool has_diar_;
    bool show_speech_;
    std::function<const char*(int)> label_;
    std::vector<Item> pending_;
};

} // namespace pk
