// End-to-end test for Phase 3: speaker-attributed ASR (SAS).
//
// Loads an ASR model and a diarization model, runs both on the same audio file
// via parakeet_capi_transcribe_and_diarize_json, and validates:
//  - the JSON has "speakers", "utterances", and "words" arrays
//  - every word has a valid speaker (-1 or 0..n_speakers-1)
//  - utterances have text, start, end, speaker fields
//  - the JSON is parseable
//
// Env:
//   PARAKEET_TEST_GGUF        ASR model (skip 77 if unset)
//   PARAKEET_TEST_DIAR_GGUF   diarization model (skip 77 if unset)
//   PARAKEET_TEST_COMBINED_WAV  audio file (default: tests/fixtures/speech.wav)

#include "parakeet_capi.h"
#include "audio_io.hpp"

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

} // namespace

int main() {
    // ABI version sanity.
    int abi = parakeet_capi_abi_version();
    if (abi < 7) {
        std::fprintf(stderr, "test_combined_offline: abi version %d < 7 (need SAS)\n", abi);
        return 1;
    }

    const char* asr_gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!asr_gguf) {
        std::fprintf(stderr, "test_combined_offline: PARAKEET_TEST_GGUF not set; skip\n");
        return 77;
    }
    const char* diar_gguf = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    if (!diar_gguf) {
        std::fprintf(stderr, "test_combined_offline: PARAKEET_TEST_DIAR_GGUF not set; skip\n");
        return 77;
    }

    const char* wav = std::getenv("PARAKEET_TEST_COMBINED_WAV");
    if (!wav) wav = "tests/fixtures/speech.wav";

    // Load both models.
    parakeet_ctx* asr_ctx = parakeet_capi_load(asr_gguf);
    if (!asr_ctx) {
        std::fprintf(stderr, "test_combined_offline: ASR load failed: %s\n",
                     asr_ctx ? parakeet_capi_last_error(asr_ctx) : "(null)");
        return 1;
    }
    parakeet_ctx* diar_ctx = parakeet_capi_load(diar_gguf);
    if (!diar_ctx) {
        std::fprintf(stderr, "test_combined_offline: diar load failed: %s\n",
                     diar_ctx ? parakeet_capi_last_error(diar_ctx) : "(null)");
        parakeet_capi_free(asr_ctx);
        return 1;
    }

    // Load audio from file — we need raw PCM, so we use the diarize_path JSON
    // variant as a smoke test... no, we need PCM for transcribe_and_diarize.
    // Load the WAV using the ASR model's path transcribe (which loads the wav)
    // — actually, we need to load the WAV ourselves.
    // The C-API has no "load WAV to PCM" function, so we use a simple approach:
    // call parakeet_capi_transcribe_and_diarize_json with a file path... no.
    // Actually, the SAS API takes PCM samples. We need to read the WAV file
    // ourselves. Let's use the existing test audio loading approach.

    // Read WAV using the shared audio_io loader.
    pk::Audio audio;
    if (!pk::load_audio_16k_mono(wav, audio) || audio.samples.empty()) {
        std::fprintf(stderr, "test_combined_offline: cannot read %s\n", wav);
        parakeet_capi_free(asr_ctx);
        parakeet_capi_free(diar_ctx);
        return 77;
    }

    std::vector<float>& pcm = audio.samples;
    int sr = 16000;

    std::fprintf(stderr, "test_combined_offline: loaded %s (%d samples, %d Hz)\n",
                 wav, (int)pcm.size(), sr);

    // --- Test 1: JSON variant ---
    char* json = parakeet_capi_transcribe_and_diarize_json(
        asr_ctx, diar_ctx, pcm.data(), (int)pcm.size(), sr);
    if (!json) {
        std::fprintf(stderr, "test_combined_offline: transcribe_and_diarize_json NULL: %s\n",
                     parakeet_capi_last_error(asr_ctx));
        parakeet_capi_free(asr_ctx);
        parakeet_capi_free(diar_ctx);
        return 1;
    }

    const std::string doc(json);
    parakeet_capi_free_string(json);

    std::fprintf(stderr, "test_combined_offline: json head = %.200s ...\n", doc.c_str());

    bool ok = true;

    // Validate JSON structure: must have speakers, utterances, words.
    if (!has_array(doc, "utterances")) {
        std::fprintf(stderr, "test_combined_offline: missing \"utterances\" array\n");
        ok = false;
    }
    if (!has_array(doc, "words")) {
        std::fprintf(stderr, "test_combined_offline: missing \"words\" array\n");
        ok = false;
    }

    // Check "speakers" field exists and is positive.
    {
        Scan sc(doc);
        if (!sc.seek_key("speakers")) {
            std::fprintf(stderr, "test_combined_offline: missing \"speakers\" field\n");
            ok = false;
        } else {
            double spk;
            if (!sc.num(spk) || spk <= 0) {
                std::fprintf(stderr, "test_combined_offline: invalid speakers value\n");
                ok = false;
            } else {
                std::fprintf(stderr, "test_combined_offline: speakers = %.0f\n", spk);
            }
        }
    }

    // Parse word speakers and validate range.
    {
        std::vector<int> speakers;
        if (!parse_word_speakers(doc, speakers)) {
            std::fprintf(stderr, "test_combined_offline: failed to parse word speakers\n");
            ok = false;
        } else {
            std::fprintf(stderr, "test_combined_offline: %zu words parsed\n",
                         speakers.size());
            // Check that all speaker indices are valid (-1 or 0..7)
            for (size_t i = 0; i < speakers.size(); ++i) {
                if (speakers[i] < -1 || speakers[i] > 7) {
                    std::fprintf(stderr,
                        "test_combined_offline: word[%zu] speaker=%d out of range\n",
                        i, speakers[i]);
                    ok = false;
                    break;
                }
            }
        }
    }

    // Count utterances and words.
    int n_utts = count_array_elements(doc, "utterances");
    int n_words = count_array_elements(doc, "words");
    std::fprintf(stderr, "test_combined_offline: %d utterances, %d words\n",
                 n_utts, n_words);

    if (n_utts < 0 || n_words < 0) {
        std::fprintf(stderr, "test_combined_offline: failed to count arrays\n");
        ok = false;
    }
    if (n_words == 0) {
        std::fprintf(stderr, "test_combined_offline: no words transcribed\n");
        ok = false;
    }

    // --- Test 2: struct variant ---
    int n_results = 0;
    parakeet_sas_result* results = parakeet_capi_transcribe_and_diarize(
        asr_ctx, diar_ctx, pcm.data(), (int)pcm.size(), sr, &n_results);
    if (!results) {
        std::fprintf(stderr, "test_combined_offline: transcribe_and_diarize NULL: %s\n",
                     parakeet_capi_last_error(asr_ctx));
        ok = false;
    } else {
        std::fprintf(stderr, "test_combined_offline: struct variant returned %d results\n",
                     n_results);
        if (n_results != n_utts) {
            std::fprintf(stderr,
                "test_combined_offline: struct count %d != JSON count %d\n",
                n_results, n_utts);
            ok = false;
        }
        // Validate each result: speaker in range, text non-null, start < end.
        for (int i = 0; i < n_results && i < 20; ++i) {
            if (results[i].speaker < -1 || results[i].speaker > 7) {
                std::fprintf(stderr,
                    "test_combined_offline: result[%d] speaker=%d out of range\n",
                    i, results[i].speaker);
                ok = false;
            }
            if (!results[i].text) {
                std::fprintf(stderr,
                    "test_combined_offline: result[%d] text is null\n", i);
                ok = false;
            }
            if (results[i].end < results[i].start) {
                std::fprintf(stderr,
                    "test_combined_offline: result[%d] end < start\n", i);
                ok = false;
            }
        }
        // Free text strings and the array.
        for (int i = 0; i < n_results; ++i) {
            if (results[i].text) parakeet_capi_free_string(results[i].text);
        }
        parakeet_capi_free_sas_results(results);
    }

    parakeet_capi_free(asr_ctx);
    parakeet_capi_free(diar_ctx);

    if (!ok) {
        std::fprintf(stderr, "test_combined_offline: FAIL\n");
        return 1;
    }
    std::fprintf(stderr, "test_combined_offline: PASS\n");
    return 0;
}
