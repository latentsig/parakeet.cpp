// VAD-path decode in groups must equal the sequential decode of the same segments,
// bit for bit: the segments are encoded one by one and decoded in groups of up to
// 16 with the exact batched decode (CPU), so text, words, token ids, frames,
// spans and confidences match what decoding each segment alone gives.
// The reference here rebuilds the segments with segment_by_vad, decodes each with
// transcribe_pcm / transcribe_with_timestamps, and adds the segment offsets the
// way the VAD path does. A small max_seg_sec gives more than 16 segments, so more
// than one group runs. Also checks that audio of at most max_seg_sec takes the
// plain path.
// Models: PARAKEET_TEST_GGUF_ULTRA, PARAKEET_TEST_GGUF_REDUX_KEEP,
// PARAKEET_TEST_GGUF_REDUX_DEQ (each needs a VAD head). Skips (77) if none is set.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "audio_io.hpp"
#include "model.hpp"

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static bool same_f(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

static bool same_tr(const Transcription& a, const Transcription& b) {
    if (a.text != b.text || a.words.size() != b.words.size() || a.tokens.size() != b.tokens.size())
        return false;
    for (size_t i = 0; i < a.words.size(); ++i)
        if (a.words[i].text != b.words[i].text || !same_f(a.words[i].start, b.words[i].start) ||
            !same_f(a.words[i].end, b.words[i].end) || !same_f(a.words[i].conf, b.words[i].conf))
            return false;
    for (size_t i = 0; i < a.tokens.size(); ++i)
        if (a.tokens[i].id != b.tokens[i].id || a.tokens[i].frame != b.tokens[i].frame ||
            a.tokens[i].span != b.tokens[i].span || !same_f(a.tokens[i].conf, b.tokens[i].conf))
            return false;
    return true;
}

static void check_model(const char* path, const Audio& clip) {
    std::fprintf(stderr, "model %s\n", path);
    auto m = Model::load(path);
    if (!m) { CHECK(false); return; }
    if (!m->config().vad.present) { std::fprintf(stderr, "  no VAD head; skip\n"); return; }
    const ParakeetConfig& cfg = m->config();
    const double enc_frame_sec =
        (double)cfg.hop_length * (double)cfg.subsampling_factor / (double)cfg.sample_rate;
    const double total = (double)clip.samples.size() / 16000.0;

    // Trim 0.3 (the default) and 0 (the cuts as they were before trimming).
    for (double trim : {0.3, 0.0})
    for (double max_seg : {3.0, 7.0}) {
        SegmenterOpts opts;
        opts.max_seg_sec = max_seg;
        opts.trim_sec = trim;
        SegmenterOpts so = opts;
        so.frame_sec = cfg.vad.frame_sec;
        const std::vector<VadSegment> segs = segment_by_vad(m->vad_probabilities(clip.samples), total, so);
        std::fprintf(stderr, "  trim %.1f max_seg %.0f s: %zu segments\n", trim, max_seg, segs.size());
        if (max_seg == 3.0) CHECK(segs.size() > 16);  // more than one group of 16
        CHECK(!segs.empty());

        // Sequential reference: each segment alone, as the VAD path used to do it.
        std::string ref_text;
        Transcription ref;
        const size_t n = clip.samples.size();
        auto at = [&](double sec) {
            const long long v = std::llround(sec * 16000.0);
            return (size_t)std::min<long long>(std::max<long long>(v, 0), (long long)n);
        };
        for (const VadSegment& sg : segs) {
            const size_t a = at(sg.start), b = std::max(a, at(sg.end));
            std::vector<float> pcm(clip.samples.begin() + (std::ptrdiff_t)a,
                                   clip.samples.begin() + (std::ptrdiff_t)b);
            if (pcm.size() < 3200) pcm.resize(3200, 0.0f);
            const double start_sec = sg.start;
            const int start_frame = (int)std::llround(sg.start / enc_frame_sec);
            const std::string t = m->transcribe_pcm(pcm, 16000, Decoder::kDefault);
            if (!t.empty()) { if (!ref_text.empty()) ref_text += ' '; ref_text += t; }
            Transcription tr = m->transcribe_with_timestamps(pcm, 16000, Decoder::kDefault);
            for (Word& w : tr.words) { w.start += (float)start_sec; w.end += (float)start_sec; }
            for (TokenInfo& k : tr.tokens) k.frame += start_frame;
            if (!tr.text.empty()) { if (!ref.text.empty()) ref.text += ' '; ref.text += tr.text; }
            ref.words.insert(ref.words.end(), tr.words.begin(), tr.words.end());
            ref.tokens.insert(ref.tokens.end(), tr.tokens.begin(), tr.tokens.end());
        }
        CHECK(!ref.tokens.empty());

        const std::string got_text = m->transcribe_pcm_vad(clip.samples, 16000, Decoder::kDefault, "", opts);
        const Transcription got = m->transcribe_pcm_vad_with_timestamps(clip.samples, 16000,
                                                                        Decoder::kDefault, "", opts);
        const bool text_ok = got_text == ref_text, ts_ok = same_tr(got, ref);
        std::fprintf(stderr, "  text %s, timestamps/tokens %s (%zu tokens)\n",
                     text_ok ? "equal" : "DIFFERENT", ts_ok ? "equal" : "DIFFERENT", ref.tokens.size());
        CHECK(text_ok);
        CHECK(ts_ok);
    }

    // Audio of at most max_seg_sec keeps the plain path.
    std::vector<float> shortclip(clip.samples.begin(), clip.samples.begin() + 16000 * 8);
    CHECK(m->transcribe_pcm_vad(shortclip, 16000) == m->transcribe_pcm(shortclip, 16000));
    CHECK(same_tr(m->transcribe_pcm_vad_with_timestamps(shortclip, 16000),
                  m->transcribe_with_timestamps(shortclip, 16000)));
}

int main() {
    std::vector<const char*> paths;
    for (const char* e : {"PARAKEET_TEST_GGUF_ULTRA", "PARAKEET_TEST_GGUF_REDUX_KEEP",
                          "PARAKEET_TEST_GGUF_REDUX_DEQ"})
        if (const char* v = std::getenv(e)) paths.push_back(v);
    if (paths.empty()) { std::puts("skip: no model env set"); return 77; }
    Audio clip;
    for (const char* f : {"tests/fixtures/two_speakers.wav", "tests/fixtures/speech.wav",
                          "tests/fixtures/two_speakers.wav", "tests/fixtures/speech.wav"}) {
        Audio x;
        if (!load_audio_16k_mono(f, x)) { std::fprintf(stderr, "wav load failed: %s\n", f); return 1; }
        clip.samples.insert(clip.samples.end(), x.samples.begin(), x.samples.end());
    }
    clip.sample_rate = 16000;
    for (const char* p : paths) check_model(p, clip);
    return failures ? 1 : 0;
}
