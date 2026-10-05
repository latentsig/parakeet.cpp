// VAD segment trimming and the word filter on real models.
//
// Clip: fixtures joined by long stretches of digital silence, so the VAD finds
// pauses. Checks, per model:
//   * trimmed segments carry less audio, and the words and their times match the
//     untrimmed decode of the same speech (offsets are right);
//   * trim 0 gives the same Transcription as before trimming existed (the
//     segments of the legacy segmenter, decoded one by one);
//   * the word filter off is byte-identical: Model call, C-API JSON;
//   * the filter on: guard member, words removed from text, words and tokens.
// Env (each optional; skip 77 when none is set):
//   PARAKEET_TEST_GGUF_ULTRA / _REDUX_KEEP / _REDUX_DEQ  models with a VAD head
//   PARAKEET_TEST_GGUF         any ASR model (with PARAKEET_TEST_SILERO_GGUF: Silero cuts)
//   PARAKEET_TEST_GGUF_CTC     a CTC model: the filter keys on the CTC head
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "audio_io.hpp"
#include "model.hpp"
#include "parakeet_capi.h"
#include "silero_vad.hpp"

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static std::vector<std::string> words_of(const std::string& s) {
    std::istringstream is(s);
    std::vector<std::string> w;
    for (std::string x; is >> x;) {
        std::string y;
        for (char c : x) if (std::isalnum((unsigned char)c)) y += (char)std::tolower((unsigned char)c);
        if (!y.empty()) w.push_back(y);
    }
    return w;
}

static bool same_f(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }
static bool same_tr(const Transcription& a, const Transcription& b) {
    if (a.text != b.text || a.words.size() != b.words.size() || a.tokens.size() != b.tokens.size() ||
        a.dropped_words != b.dropped_words)
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

// speech.wav, 24 s of digital silence, speech.wav, 24 s of silence: about 62 s.
static Audio make_clip(double* first_end, double* second_start) {
    Audio sp;
    CHECK(load_audio_16k_mono("tests/fixtures/speech.wav", sp));
    Audio clip;
    clip.sample_rate = 16000;
    const std::vector<float> gap((size_t)(24 * 16000), 0.0f);
    clip.samples.insert(clip.samples.end(), sp.samples.begin(), sp.samples.end());
    *first_end = (double)clip.samples.size() / 16000.0;
    clip.samples.insert(clip.samples.end(), gap.begin(), gap.end());
    *second_start = (double)clip.samples.size() / 16000.0;
    clip.samples.insert(clip.samples.end(), sp.samples.begin(), sp.samples.end());
    clip.samples.insert(clip.samples.end(), gap.begin(), gap.end());
    return clip;
}

static size_t edit_distance(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    std::vector<size_t> prev(b.size() + 1), cur(b.size() + 1);
    for (size_t j = 0; j <= b.size(); ++j) prev[j] = j;
    for (size_t i = 1; i <= a.size(); ++i) {
        cur[0] = i;
        for (size_t j = 1; j <= b.size(); ++j)
            cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
        std::swap(prev, cur);
    }
    return prev[b.size()];
}

static double covered(const std::vector<VadSegment>& s) {
    double t = 0.0;
    for (const auto& g : s) t += g.end - g.start;
    return t;
}

// `fn` set: Silero probabilities (opts are Silero's); else the model's own head.
static void check_trim(const Model& m, const Audio& clip, double first_end, double second_start,
                       SegmenterOpts opts, const Model::VadProbabilityFn* fn, const char* what, bool strict,
                       bool quiet_gap) {
    std::fprintf(stderr, "  %s: trim\n", what);
    const double total = (double)clip.samples.size() / 16000.0;
    SegmenterOpts so = opts;
    if (!fn) so.frame_sec = m.config().vad.frame_sec;
    const std::vector<float> p = fn ? (*fn)(clip.samples) : m.vad_probabilities(clip.samples);
    SegmenterOpts old = so;
    old.trim_sec = 0.0;
    const auto segs_new = segment_by_vad(p, total, so);
    const auto segs_old = segment_by_vad(p, total, old);
    std::fprintf(stderr, "    %zu segments, %.1f s with trim, %.1f s without\n", segs_new.size(),
                 covered(segs_new), covered(segs_old));
    CHECK(segs_new.size() >= 2 && segs_new.size() == segs_old.size());
    CHECK(covered(segs_new) < covered(segs_old));
    // The detector calls the digital silence between the speech non-speech: only
    // the speech is decoded. (The Redux head fires on digital silence, so it does
    // not get this check.)
    if (quiet_gap) {
        CHECK(covered(segs_new) < covered(segs_old) - 10.0);
        CHECK(covered(segs_new) < 2.0 * first_end + 4 * so.trim_sec + 1.0);
    }

    // Word times: the second copy of the speech is the first one shifted.
    opts.trim_sec = so.trim_sec;
    const Transcription tr = m.transcribe_pcm_vad_with_timestamps(clip.samples, 16000, Decoder::kDefault, "", opts, fn);
    opts.trim_sec = 0.0;
    const Transcription tr0 = m.transcribe_pcm_vad_with_timestamps(clip.samples, 16000, Decoder::kDefault, "", opts, fn);
    CHECK(tr.dropped_words == -1 && tr0.dropped_words == -1);
    CHECK(!tr.words.empty());
    // Another cut changes the audio the decoder sees, so a model may differ by a
    // word or two; the head model with these fixtures does not.
    if (strict) {
        CHECK(words_of(tr.text) == words_of(tr0.text));
        CHECK(tr.words.size() == tr0.words.size());
    } else {
        const auto a = words_of(tr.text), b = words_of(tr0.text);
        CHECK(edit_distance(a, b) <= 1 + b.size() / 10);
    }
    float prev = -1.0f;
    for (const Word& w : tr.words) {
        CHECK(w.start >= prev - 1e-3f && w.end >= w.start && w.end <= (float)total + 0.01f);
        prev = w.start;
    }
    for (size_t i = 1; i < tr.tokens.size(); ++i) CHECK(tr.tokens[i].frame >= tr.tokens[i - 1].frame);
    for (size_t i = 0; strict && i < tr.words.size() && i < tr0.words.size(); ++i) {
        // Trimming moves the start of a segment, and with it the 80 ms frame grid
        // of the encoder: times agree to a few frames.
        CHECK(std::fabs(tr.words[i].start - tr0.words[i].start) < 0.5f);
    }
    // The first words are in the first speech, the last words in the second.
    if (!tr.words.empty()) {
        CHECK(tr.words.front().start < (float)first_end);
        CHECK(tr.words.back().start > (float)second_start);
        size_t in_gap = 0;
        for (const Word& w : tr.words) in_gap += (w.start > (float)first_end + 0.5f && w.start < (float)second_start - 0.5f);
        if (quiet_gap) CHECK(in_gap == 0);
    }
    // The text path matches the timestamp path.
    CHECK(m.transcribe_pcm_vad(clip.samples, 16000, Decoder::kDefault, "", opts, fn) == tr0.text);
}

static void check_filter(const Model& m, const char* wav_path, const Audio& clip, const SegmenterOpts& opts,
                         const Model::VadProbabilityFn* fn, const char* what) {
    std::fprintf(stderr, "  %s: filter\n", what);
    // Off: the filter object changes nothing, on a short clip and on a long one.
    Audio sp;
    CHECK(load_audio_16k_mono("tests/fixtures/speech.wav", sp));
    {
        const Transcription plain = m.transcribe_with_timestamps(sp.samples, 16000, Decoder::kDefault);
        const Transcription off = m.transcribe_pcm_vad_with_timestamps(sp.samples, 16000, Decoder::kDefault, "", opts, fn, WordFilter());
        CHECK(same_tr(plain, off) && off.dropped_words == -1);
    }
    const Transcription base = m.transcribe_pcm_vad_with_timestamps(clip.samples, 16000, Decoder::kDefault, "", opts, fn);
    CHECK(same_tr(base, m.transcribe_pcm_vad_with_timestamps(clip.samples, 16000, Decoder::kDefault, "", opts, fn, WordFilter())));
    CHECK(!base.words.empty());

    // A mild filter keeps clean speech as it is; the guard member says nothing was dropped.
    WordFilter mild;
    mild.min_local_conf = 0.5f;
    const Transcription kept = m.transcribe_pcm_vad_with_timestamps(clip.samples, 16000, Decoder::kDefault, "", opts, fn, mild);
    CHECK(kept.dropped_words == 0);
    Transcription expect = base;
    expect.dropped_words = 0;
    CHECK(same_tr(kept, expect));
    // The text path with a filter gives the filtered text.
    CHECK(m.transcribe_pcm_vad(clip.samples, 16000, Decoder::kDefault, "", opts, fn, mild) == kept.text);

    // A threshold of 1 drops every word (no word has mean confidence 1), per
    // decode unit, and the count adds up.
    WordFilter all;
    all.min_local_conf = 1.0f;
    const Transcription none = m.transcribe_pcm_vad_with_timestamps(clip.samples, 16000, Decoder::kDefault, "", opts, fn, all);
    CHECK(none.words.empty() && none.tokens.empty() && none.text.empty());
    CHECK(none.dropped_words == (int)base.words.size());
    // A radius too small to reach a neighbour leaves the own confidence: the
    // low words go, the rest stays, and the numbers agree between words and tokens.
    WordFilter own;
    own.min_local_conf = 0.9f;
    own.local_radius_sec = 0.001f;
    const Transcription some = m.transcribe_pcm_vad_with_timestamps(clip.samples, 16000, Decoder::kDefault, "", opts, fn, own);
    CHECK(some.dropped_words >= 0 && (size_t)some.dropped_words + some.words.size() == base.words.size());
    CHECK(some.tokens.size() <= base.tokens.size());
    for (const Word& w : some.words) CHECK(w.conf >= 0.9f - 1e-6f);
    CHECK(words_of(some.text).size() <= base.words.size());
    (void)wav_path;
}

static char* take_dup(char* p, std::string& out) {
    if (p) { out = p; parakeet_capi_free_string(p); }
    return p;
}

static void check_capi(const char* asr_path, const char* silero_path, const char* what) {
    std::fprintf(stderr, "  %s: C-API\n", what);
    parakeet_ctx* ctx = parakeet_capi_load(asr_path);
    CHECK(ctx != nullptr);
    if (!ctx) return;
    const char* wav = "tests/fixtures/speech.wav";
    std::string plain, off, none_opts, kept, dropped;
    CHECK(take_dup(parakeet_capi_transcribe_path_json(ctx, wav, 0), plain));
    // No options, empty options, and an explicit off give the plain document, byte for byte.
    CHECK(take_dup(parakeet_capi_transcribe_path_json_with(ctx, wav, 0, nullptr), off) && off == plain);
    CHECK(take_dup(parakeet_capi_transcribe_path_json_with(ctx, wav, 0, ""), none_opts) && none_opts == plain);
    CHECK(take_dup(parakeet_capi_transcribe_path_json_with(ctx, wav, 0, "{}"), none_opts) && none_opts == plain);
    CHECK(take_dup(parakeet_capi_transcribe_path_json_with(ctx, wav, 0, "{\"min_local_conf\":0,\"local_radius\":3}"), none_opts) &&
          none_opts == plain);
    CHECK(plain.find("\"guard\"") == std::string::npos);
    // On: guard present; 0.5 keeps clean speech; 1 drops every word.
    CHECK(take_dup(parakeet_capi_transcribe_path_json_with(ctx, wav, 0, "{\"min_local_conf\":0.5}"), kept));
    CHECK(kept.size() > plain.size() && kept.find("\"guard\":{\"dropped_words\":0}") != std::string::npos);
    CHECK(kept.substr(0, plain.size() - 1) == plain.substr(0, plain.size() - 1));
    CHECK(take_dup(parakeet_capi_transcribe_path_json_with(ctx, wav, 0, "{\"min_local_conf\":1}"), dropped));
    CHECK(dropped.find("\"text\":\"\"") != std::string::npos && dropped.find("\"words\":[]") != std::string::npos &&
          dropped.find("\"tokens\":[]") != std::string::npos && dropped.find("\"guard\":{\"dropped_words\":") != std::string::npos &&
          dropped.find("\"dropped_words\":0}") == std::string::npos);
    // Errors: NULL, bad keys, bad values, the last error says which.
    CHECK(parakeet_capi_transcribe_path_json_with(nullptr, wav, 0, nullptr) == nullptr);
    CHECK(parakeet_capi_transcribe_path_json_with(ctx, nullptr, 0, nullptr) == nullptr);
    CHECK(parakeet_capi_transcribe_path_json_with(ctx, "/nonexistent.wav", 0, nullptr) == nullptr);
    const struct { const char* json; const char* word; } bad[] = {
        {"{\"min_local_conf\":1.5}", "min_local_conf"}, {"{\"min_local_conf\":-0.1}", "min_local_conf"},
        {"{\"local_radius\":0}", "local_radius"}, {"{\"drop_punct_only\":1}", "drop_punct_only"},
        {"{\"threshold\":0.5}", "threshold"},  // a VAD key is not a filter key
        {"{\"nope\":1}", "nope"}, {"[1]", "JSON object"}};
    for (const auto& b : bad) {
        CHECK(parakeet_capi_transcribe_path_json_with(ctx, wav, 0, b.json) == nullptr);
        CHECK(std::string(parakeet_capi_last_error(ctx)).find(b.word) != std::string::npos);
    }
    // The VAD variant takes the same filter keys and the trim key.
    if (silero_path) {
        parakeet_ctx* v = parakeet_capi_load(silero_path);
        CHECK(v != nullptr);
        if (v) {
            std::string a, b, c;
            CHECK(take_dup(parakeet_capi_transcribe_path_json_vad_with(ctx, v, wav, 0, nullptr), a));
            CHECK(take_dup(parakeet_capi_transcribe_path_json_vad_with(ctx, v, wav, 0, "{\"trim\":0,\"max_segment\":10}"), b));
            CHECK(take_dup(parakeet_capi_transcribe_path_json_vad_with(ctx, v, wav, 0, "{\"trim\":0.1,\"max_segment\":10,\"min_local_conf\":0.5,\"drop_punct_only\":true}"), c));
            CHECK(a.find("\"guard\"") == std::string::npos && b.find("\"guard\"") == std::string::npos &&
                  c.find("\"guard\":{\"dropped_words\":") != std::string::npos);
            CHECK(parakeet_capi_transcribe_path_json_vad_with(ctx, v, wav, 0, "{\"trim\":-1}") == nullptr);
            CHECK(std::string(parakeet_capi_last_error(ctx)).find("trim") != std::string::npos);
            CHECK(parakeet_capi_transcribe_path_json_vad_with(ctx, v, wav, 0, "{\"min_local_conf\":2}") == nullptr);
            parakeet_capi_free(v);
        }
    }
    parakeet_capi_free(ctx);
}

int main() {
    // The Redux head fires on digital silence, so only Ultra gets the "silence is not decoded" checks.
    std::vector<std::pair<const char*, bool>> heads;
    for (const char* e : {"PARAKEET_TEST_GGUF_ULTRA", "PARAKEET_TEST_GGUF_REDUX_KEEP", "PARAKEET_TEST_GGUF_REDUX_DEQ"})
        if (const char* v = std::getenv(e)) heads.push_back({v, std::strstr(e, "ULTRA") != nullptr});
    const char* any = std::getenv("PARAKEET_TEST_GGUF");
    const char* silero = std::getenv("PARAKEET_TEST_SILERO_GGUF");
    const char* ctc = std::getenv("PARAKEET_TEST_GGUF_CTC");
    if (heads.empty() && !(any && silero) && !ctc) { std::puts("skip: no model env set"); return 77; }

    double first_end = 0.0, second_start = 0.0;
    const Audio clip = make_clip(&first_end, &second_start);

    for (const auto& h : heads) {
        const char* path = h.first;
        std::fprintf(stderr, "head model %s\n", path);
        std::unique_ptr<Model> m = Model::load(path);
        if (!m) { CHECK(false); continue; }
        if (!m->config().vad.present) continue;
        check_trim(*m, clip, first_end, second_start, SegmenterOpts(), nullptr, "head", true, h.second);
        check_filter(*m, "", clip, SegmenterOpts(), nullptr, "head");
        check_capi(path, silero, "head");
    }
    if (any && silero) {
        std::fprintf(stderr, "silero cuts, model %s\n", any);
        std::string err;
        std::unique_ptr<SileroVad> sv = SileroVad::load(silero, &err);
        std::unique_ptr<Model> m = Model::load(any);
        CHECK(sv && m);
        if (sv && m) {
            const SileroVad* svp = sv.get();
            const Model::VadProbabilityFn fn = [svp](const std::vector<float>& pcm) {
                return svp->probabilities(pcm.data(), pcm.size(), 16000);
            };
            const SegmenterOpts so = default_segmenter_opts(VadKind::kSilero);
            check_trim(*m, clip, first_end, second_start, so, &fn, "silero", false, true);
            check_filter(*m, "", clip, so, &fn, "silero");
            check_capi(any, silero, "silero");
        }
    }
    if (ctc) {
        std::fprintf(stderr, "CTC model %s\n", ctc);
        std::unique_ptr<Model> m = Model::load(ctc);
        CHECK(m != nullptr);
        if (m) {
            Audio sp;
            CHECK(load_audio_16k_mono("tests/fixtures/speech.wav", sp));
            Transcription t = m->transcribe_with_timestamps(sp.samples, 16000, Decoder::kCTC);
            const Transcription ref = t;
            WordFilter f;
            f.min_local_conf = 0.5f;
            f.drop_punct_only = true;
            apply_word_filter(t, f);
            CHECK(t.dropped_words == 0 && t.text == ref.text && t.words.size() == ref.words.size());
            f.min_local_conf = 1.0f;
            apply_word_filter(t, f);
            CHECK(t.words.empty() && t.tokens.empty() && t.dropped_words == (int)ref.words.size());
        }
    }
    if (failures) return 1;
    std::puts("test_vad_trim_filter: OK");
    return 0;
}
