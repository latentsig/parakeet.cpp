// Speaker-attributed ASR (SAS) and streaming diarization through the C-API.
//
// Runs on tests/fixtures/two_speakers.wav (LibriSpeech speakers 1272 and 2086
// alternating A-B-A-B) and checks:
//  1. transcribe_and_diarize_json: valid document, every word attributed to a
//     speaker, speaker turns follow A-B-A-B
//  2. transcribe_and_diarize (struct): same utterances as the JSON variant
//  3. diarize_stream_*: fed live in 0.5 s pieces, the segments match the
//     offline diarize_pcm segments (speaker, boundaries within 0.1 s)
//  4. sas_stream_*: fed live in 0.5 s pieces, same turn pattern and about the
//     same words as the offline SAS
//
// Env: PARAKEET_TEST_GGUF (ASR model) + PARAKEET_TEST_DIAR_GGUF; skips (77)
// when either is unset. WORKING_DIRECTORY is the repo root.

#include "parakeet_capi.h"
#include "audio_io.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Reuse the tiny JSON scanner pattern from test_capi_timestamps.cpp.
namespace {

struct Scan {
    const std::string& s;
    size_t i = 0;
    explicit Scan(const std::string& str) : s(str) {}
    void ws() { while (i < s.size() && (s[i]==' '||s[i]=='\t'||s[i]=='\n'||s[i]=='\r')) ++i; }
    bool eat(char c) { ws(); if (i < s.size() && s[i]==c) { ++i; return true; } return false; }
    bool str(std::string& out) {
        ws();
        if (i >= s.size() || s[i] != '"') return false;
        ++i; out.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                char c = s[i+1];
                switch (c) {
                    case 'n': out += '\n'; break; case 't': out += '\t'; break;
                    case 'r': out += '\r'; break; case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break; case '"': out += '"'; break;
                    case '\\': out += '\\'; break; case '/': out += '/'; break;
                    default: out += c; break;
                }
                i += 2;
            } else { out += s[i++]; }
        }
        if (i >= s.size()) return false;
        ++i; return true;
    }
    bool num(double& out) {
        ws();
        size_t st = i;
        while (i < s.size() && std::strchr("+-0123456789.eE", s[i])) ++i;
        if (i == st) return false;
        out = std::strtod(s.substr(st, i - st).c_str(), nullptr);
        return true;
    }
    bool seek_key(const char* key) {
        std::string pat = std::string("\"") + key + "\"";
        size_t p = s.find(pat, i);
        if (p == std::string::npos) return false;
        i = p + pat.size();
        return eat(':');
    }
};

// Check that a JSON array key exists (e.g. "utterances":[...]).
bool has_array(const std::string& s, const char* key) {
    Scan sc(s);
    return sc.seek_key(key) && sc.eat('[');
}

// Count elements in an array (rough: count top-level '{' or ',' at depth 1).
int count_array_elements(const std::string& s, const char* key) {
    Scan sc(s);
    if (!sc.seek_key(key)) return -1;
    if (!sc.eat('[')) return -1;
    sc.ws();
    if (sc.i < s.size() && s[sc.i] == ']') return 0;
    int count = 0;
    int depth = 0;
    while (sc.i < s.size()) {
        char c = s[sc.i];
        if (c == '{') { if (depth == 0) ++count; ++depth; }
        else if (c == '}') { --depth; }
        else if (c == ']' && depth == 0) break;
        ++sc.i;
    }
    return count;
}

// Parse all "speaker" integer values from the words array.
bool parse_word_speakers(const std::string& s, std::vector<int>& speakers) {
    Scan sc(s);
    if (!sc.seek_key("words")) return false;
    if (!sc.eat('[')) return false;
    sc.ws();
    if (sc.i < s.size() && s[sc.i] == ']') return true;  // empty

    while (true) {
        if (!sc.eat('{')) return false;
        // Parse fields until '}'
        while (true) {
            std::string key;
            if (!sc.str(key)) return false;
            if (!sc.eat(':')) return false;
            if (key == "speaker") {
                double v;
                if (!sc.num(v)) return false;
                speakers.push_back((int)v);
            } else {
                // Skip value: string or number
                std::string tmp;
                double d;
                if (!sc.str(tmp) && !sc.num(d)) return false;
            }
            if (sc.eat(',')) continue;
            break;
        }
        if (!sc.eat('}')) return false;
        if (sc.eat(',')) continue;
        break;
    }
    return sc.eat(']');
}

// Collapse consecutive repeats: [0,0,1,0,0,1] -> [0,1,0,1].
std::vector<int> turns(const std::vector<int>& spk) {
    std::vector<int> t;
    for (int s : spk) if (t.empty() || t.back() != s) t.push_back(s);
    return t;
}

std::string show(const std::vector<int>& v) {
    std::string s;
    for (int x : v) s += (s.empty() ? "" : ",") + std::to_string(x);
    return "[" + s + "]";
}

int word_count(const char* text) {
    int n = 0;
    bool in = false;
    for (const char* p = text; *p; ++p) {
        const bool sp = *p == ' ';
        if (!sp && !in) ++n;
        in = !sp;
    }
    return n;
}

// Offline diarize_pcm segments as (speaker, start, end) triples.
bool parse_segments(const std::string& doc, std::vector<std::array<double, 3>>& out) {
    Scan sc(doc);
    if (!sc.seek_key("segments") || !sc.eat('[')) return false;
    if (sc.eat(']')) return true;
    do {
        std::array<double, 3> seg{};
        if (!sc.eat('{')) return false;
        for (int k = 0; k < 3; ++k) {
            std::string key;
            if (!sc.str(key) || !sc.eat(':') || !sc.num(seg[k])) return false;
            if (k < 2 && !sc.eat(',')) return false;
        }
        if (!sc.eat('}')) return false;
        out.push_back(seg);
    } while (sc.eat(','));
    return sc.eat(']');
}

} // namespace

#define CHECK(cond, ...)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: " __VA_ARGS__);               \
            std::fprintf(stderr, "\n");                               \
            ok = false;                                               \
        }                                                             \
    } while (0)

int main() {
    const char* asr_gguf = std::getenv("PARAKEET_TEST_GGUF");
    const char* diar_gguf = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    if (!asr_gguf || !diar_gguf) {
        std::fprintf(stderr, "test_combined_offline: PARAKEET_TEST_GGUF and/or "
                             "PARAKEET_TEST_DIAR_GGUF not set; skip\n");
        return 77;
    }
    if (parakeet_capi_abi_version() < 7) {
        std::fprintf(stderr, "test_combined_offline: ABI < 7\n");
        return 1;
    }
    parakeet_ctx* asr = parakeet_capi_load(asr_gguf);
    parakeet_ctx* diar = parakeet_capi_load(diar_gguf);
    if (!asr || !diar) {
        std::fprintf(stderr, "test_combined_offline: load failed\n");
        parakeet_capi_free(asr);
        parakeet_capi_free(diar);
        return 1;
    }
    if (parakeet_capi_model_kind(asr) != PARAKEET_MODEL_KIND_ASR ||
        parakeet_capi_model_kind(diar) != PARAKEET_MODEL_KIND_DIARIZATION) {
        std::fprintf(stderr, "test_combined_offline: model_kind mismatch\n");
        parakeet_capi_free(asr);
        parakeet_capi_free(diar);
        return 1;
    }
    pk::Audio audio;
    if (!pk::load_audio_16k_mono("tests/fixtures/two_speakers.wav", audio)) {
        std::fprintf(stderr, "test_combined_offline: cannot read the fixture\n");
        return 1;
    }
    const std::vector<float>& pcm = audio.samples;
    const int n = (int)pcm.size();
    const std::vector<int> expected_turns = {0, 1, 0, 1};
    bool ok = true;

    // Wrong-model guards.
    CHECK(parakeet_capi_diarize_pcm(asr, pcm.data(), n, 16000) == nullptr,
          "diarize_pcm accepted an ASR context");
    CHECK(parakeet_capi_transcribe_pcm(diar, pcm.data(), n, 16000, 0) == nullptr,
          "transcribe_pcm accepted a diarization context");

    // 1. JSON variant.
    int n_words_offline = 0, n_utts_json = -1;
    {
        char* json = parakeet_capi_transcribe_and_diarize_json(asr, diar, pcm.data(), n, 16000);
        CHECK(json != nullptr, "transcribe_and_diarize_json: %s", parakeet_capi_last_error(asr));
        if (json) {
            const std::string doc(json);
            parakeet_capi_free_string(json);
            std::vector<int> spk;
            CHECK(parse_word_speakers(doc, spk), "cannot parse the words array");
            n_words_offline = (int)spk.size();
            n_utts_json = count_array_elements(doc, "utterances");
            int unassigned = 0;
            for (int s : spk) unassigned += s < 0;
            std::printf("offline SAS: %d words, %d utterances, turns %s, %d unassigned\n",
                        n_words_offline, n_utts_json, show(turns(spk)).c_str(), unassigned);
            CHECK(n_words_offline > 40, "too few words (%d)", n_words_offline);
            CHECK(unassigned == 0, "%d words without a speaker", unassigned);
            CHECK(turns(spk) == expected_turns, "turns %s, expected [0,1,0,1]",
                  show(turns(spk)).c_str());
        }
    }

    // 2. Struct variant.
    {
        parakeet_sas_result* r = nullptr;
        int nr = 0;
        const int rc = parakeet_capi_transcribe_and_diarize(asr, diar, pcm.data(), n, 16000, &r, &nr);
        CHECK(rc == 0, "transcribe_and_diarize: %s", parakeet_capi_last_error(asr));
        CHECK(nr == n_utts_json, "struct count %d != JSON count %d", nr, n_utts_json);
        for (int i = 0; i < nr; ++i)
            CHECK(r[i].text && r[i].start <= r[i].end && r[i].speaker >= 0,
                  "bad result %d", i);
        parakeet_capi_free_sas_results(r, nr);
    }

    // 3. Streaming diarization vs offline diarization.
    {
        char* json = parakeet_capi_diarize_pcm(diar, pcm.data(), n, 16000);
        std::vector<std::array<double, 3>> offline;
        CHECK(json && parse_segments(json, offline), "diarize_pcm");
        parakeet_capi_free_string(json);

        parakeet_diar_stream* ds = parakeet_capi_diarize_stream_begin(diar);
        CHECK(ds != nullptr, "diarize_stream_begin: %s", parakeet_capi_last_error(diar));
        std::vector<parakeet_diar_segment> streamed;
        for (int lo = 0; ds && lo < n; lo += 8000) {
            const int len = std::min(8000, n - lo);
            parakeet_diar_segment* segs = nullptr;
            int ns = 0;
            const int rc = parakeet_capi_diarize_stream_feed(ds, pcm.data() + lo, len,
                                                             lo + len >= n, &segs, &ns);
            CHECK(rc == 0, "diarize_stream_feed: %s", parakeet_capi_last_error(diar));
            streamed.insert(streamed.end(), segs, segs + ns);
            parakeet_capi_free_diar_segments(segs);
        }
        parakeet_capi_diarize_stream_free(ds);
        std::sort(streamed.begin(), streamed.end(), [](const auto& a, const auto& b) {
            return a.start != b.start ? a.start < b.start : a.speaker < b.speaker;
        });
        std::printf("streaming diarization: %zu segments (offline %zu)\n",
                    streamed.size(), offline.size());
        CHECK(streamed.size() == offline.size(), "segment count differs");
        for (size_t i = 0; i < std::min(streamed.size(), offline.size()); ++i) {
            std::printf("  spk%d %6.2f-%6.2f   offline spk%d %6.2f-%6.2f\n",
                        streamed[i].speaker, streamed[i].start, streamed[i].end,
                        (int)offline[i][0], offline[i][1], offline[i][2]);
            CHECK(streamed[i].speaker == (int)offline[i][0] &&
                  std::fabs(streamed[i].start - offline[i][1]) <= 0.1 &&
                  std::fabs(streamed[i].end - offline[i][2]) <= 0.1,
                  "segment %zu differs", i);
        }
    }

    // 3b. Low-latency streaming diarization: diarized time trails the audio by
    //     at most the mode's latency, and _active names the current speaker.
    {
        parakeet_diar_stream* ds = parakeet_capi_diarize_stream_begin_latency(diar, PARAKEET_DIAR_LATENCY_LOW);
        CHECK(ds != nullptr, "diarize_stream_begin_latency: %s", parakeet_capi_last_error(diar));
        const int latency = ds ? parakeet_capi_diarize_stream_chunk_samples(ds) : 0;
        CHECK(latency == 16640, "LOW latency is %d samples, expected 16640 (1.04 s)", latency);
        std::vector<parakeet_diar_segment> streamed;
        int active_checks = 0;
        float prev_t = 0.0f;
        for (int lo = 0; ds && lo < n; lo += 1600) {
            const int len = std::min(1600, n - lo);
            parakeet_diar_segment* segs = nullptr;
            int ns = 0;
            const bool last = lo + len >= n;
            CHECK(parakeet_capi_diarize_stream_feed(ds, pcm.data() + lo, len, last, &segs, &ns) == 0,
                  "feed: %s", parakeet_capi_last_error(diar));
            streamed.insert(streamed.end(), segs, segs + ns);
            parakeet_capi_free_diar_segments(segs);
            if (last) break;
            const float fed = (lo + len) / 16000.0f;
            const float t = parakeet_capi_diarize_stream_time(ds);
            CHECK(fed - t <= 1.04f + 0.24f + 1e-3f, "diarized %.2f s of %.2f s fed", t, fed);
            // Mid-utterance of each speaker the current speaker must be known,
            // checked when the diarized time first passes each probe (it
            // advances in 0.72 s steps).
            for (const float probe : {3.0f, 9.0f, 16.0f, 22.0f}) {
                if (prev_t < probe && t >= probe) {
                    parakeet_diar_segment* act = nullptr;
                    int na = 0;
                    CHECK(parakeet_capi_diarize_stream_active(ds, &act, &na) == 0, "active");
                    const int want = (probe == 3.0f || probe == 16.0f) ? 0 : 1;
                    CHECK(na == 1 && act[0].speaker == want && act[0].end == t,
                          "at %.2f s: %d active, speaker %d (want %d)", t, na,
                          na ? act[0].speaker : -1, want);
                    parakeet_capi_free_diar_segments(act);
                    ++active_checks;
                }
            }
            prev_t = t;
        }
        parakeet_capi_diarize_stream_free(ds);
        std::vector<int> spk;
        std::sort(streamed.begin(), streamed.end(), [](const auto& a, const auto& b) { return a.start < b.start; });
        for (const auto& g : streamed) spk.push_back(g.speaker);
        std::printf("low-latency diarization: %zu segments, turns %s, %d active checks\n",
                    streamed.size(), show(turns(spk)).c_str(), active_checks);
        CHECK(turns(spk) == expected_turns, "low-latency turns %s", show(turns(spk)).c_str());
        CHECK(active_checks == 4, "only %d of 4 active checks ran", active_checks);
    }

    // 4. Streaming SAS (checkpoint config, then 1.04 s latency).
    for (const int latency : {PARAKEET_DIAR_LATENCY_MODEL, PARAKEET_DIAR_LATENCY_LOW}) {
        parakeet_sas_stream* ss = parakeet_capi_sas_stream_begin_latency(asr, diar, latency);
        CHECK(ss != nullptr, "sas_stream_begin");
        std::vector<int> spk;
        int words = 0;
        std::string text;
        for (int lo = 0; ss && lo < n; lo += 8000) {
            const int len = std::min(8000, n - lo);
            parakeet_sas_result* r = nullptr;
            int nr = 0;
            const int rc = parakeet_capi_sas_stream_feed(ss, pcm.data() + lo, len,
                                                         lo + len >= n, &r, &nr);
            CHECK(rc == 0, "sas_stream_feed: %s", parakeet_capi_last_error(asr));
            for (int i = 0; i < nr; ++i) {
                spk.push_back(r[i].speaker);
                words += word_count(r[i].text);
                text += std::string(text.empty() ? "" : " ") + r[i].text;
            }
            parakeet_capi_free_sas_results(r, nr);
        }
        parakeet_capi_sas_stream_free(ss);
        std::printf("streaming SAS (latency mode %d): %d words, turns %s\n  %s\n", latency, words,
                    show(turns(spk)).c_str(), text.c_str());
        CHECK(turns(spk) == expected_turns, "streaming turns %s", show(turns(spk)).c_str());
        CHECK(std::abs(words - n_words_offline) <= 3, "streaming words %d vs offline %d",
              words, n_words_offline);
    }

    parakeet_capi_free(asr);
    parakeet_capi_free(diar);
    std::printf(ok ? "test_combined_offline: PASS\n" : "test_combined_offline: FAIL\n");
    return ok ? 0 : 1;
}
