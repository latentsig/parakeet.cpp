// Scene stream on real models: composition changes nothing.
//  1. asr+diar+tagger: utterances == sas_stream on the same PCM and pieces;
//     sounds == sound_stream alone.
//  2. scene_sound_only: tagger alone returns sounds and empty word arrays.
//  3. asr+tagger (no diar): words have speaker -1.
//  4. scene_wrong_kinds: a tagger passed as ASR is rejected with a message.
// Needs PARAKEET_TEST_GGUF, PARAKEET_TEST_DIAR_GGUF, PARAKEET_TEST_CED_GGUF.
#include "parakeet_capi.h"
#include "audio_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Load a WAV as 16 kHz mono (pk::load_audio_16k_mono downmixes and resamples).
static bool load16k(const std::string& path, std::vector<float>& x) {
    pk::Audio a;
    if (!pk::load_audio_16k_mono(path, a)) return false;
    x = std::move(a.samples);
    return true;
}

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL: " __VA_ARGS__); std::fprintf(stderr, "\n"); ++fails; } } while (0)

static std::string feed_all_scene(parakeet_scene_stream* s, const std::vector<float>& pcm, int piece) {
    std::string all;
    for (size_t i = 0; i < pcm.size(); i += piece) {
        const int n = (int)std::min<size_t>(piece, pcm.size() - i);
        char* j = parakeet_capi_scene_stream_feed_json(s, pcm.data() + i, n, i + piece >= pcm.size());
        if (!j) { std::fprintf(stderr, "feed: %s\n", parakeet_capi_scene_stream_last_error(s)); ++fails; return all; }
        all += j; all += "\n";
        parakeet_capi_free_string(j);
    }
    return all;
}
// Pull "text" values of "utterances" from the JSON lines, in order. Scans
// only inside the "utterances" array (not "words", which also has "text"),
// and un-escapes \" and \\ so a quote inside a word does not truncate early.
static std::vector<std::string> utt_texts(const std::string& jl) {
    std::vector<std::string> out;
    size_t p = 0;
    while ((p = jl.find("\"utterances\":[", p)) != std::string::npos) {
        size_t q = p + 14;
        const size_t end = jl.find("],\"words\"", q);
        while ((q = jl.find("\"text\":\"", q)) != std::string::npos && q < end) {
            q += 8;
            std::string s;
            while (q < jl.size() && jl[q] != '"') {
                if (jl[q] == '\\' && q + 1 < jl.size()) { s += jl[q + 1]; q += 2; }
                else { s += jl[q]; ++q; }
            }
            out.push_back(std::move(s));
        }
        p = end;
    }
    return out;
}

int main() {
    const char* a = std::getenv("PARAKEET_TEST_GGUF");
    const char* d = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    const char* c = std::getenv("PARAKEET_TEST_CED_GGUF");
    if (!a || !d || !c) { std::fprintf(stderr, "SKIP: needs ASR, diar and CED GGUFs\n"); return 77; }
    parakeet_ctx* asr = parakeet_capi_load(a);
    parakeet_ctx* diar = parakeet_capi_load(d);
    parakeet_ctx* tag = parakeet_capi_load(c);
    if (!asr || !diar || !tag) { std::fprintf(stderr, "FAIL: load\n"); return 1; }

    // Speech, then a rooster, then speech again.
    std::vector<float> pcm, x;
    load16k(std::string(PK_SOURCE_DIR) + "/tests/fixtures/two_speakers.wav", x);
    pcm.insert(pcm.end(), x.begin(), x.end());
    load16k(std::string(PK_CED_SOURCE_DIR) + "/benchmarks/demo/clips/rooster.wav", x);
    pcm.insert(pcm.end(), x.begin(), x.end());
    load16k(std::string(PK_SOURCE_DIR) + "/tests/fixtures/speech.wav", x);
    pcm.insert(pcm.end(), x.begin(), x.end());
    const int piece = 8000;  // 0.5 s, like test_combined_offline

    // 1a. reference utterances from sas_stream
    std::vector<std::string> ref;
    {
        parakeet_sas_stream* ss = parakeet_capi_sas_stream_begin_latency(asr, diar, PARAKEET_DIAR_LATENCY_LOW);
        for (size_t i = 0; i < pcm.size(); i += piece) {
            parakeet_sas_result* r = nullptr; int nr = 0;
            const int n = (int)std::min<size_t>(piece, pcm.size() - i);
            parakeet_capi_sas_stream_feed(ss, pcm.data() + i, n, i + piece >= pcm.size(), &r, &nr);
            for (int k = 0; k < nr; ++k) ref.push_back(r[k].text);
            parakeet_capi_free_sas_results(r, nr);
        }
        parakeet_capi_sas_stream_free(ss);
    }
    // 1b. reference sounds from sound_stream
    std::string ref_sounds;
    {
        parakeet_sound_stream* s = parakeet_capi_sound_stream_begin(tag, nullptr);
        for (size_t i = 0; i < pcm.size(); i += piece) {
            parakeet_sound_segment* o = nullptr; int no = 0;
            const int n = (int)std::min<size_t>(piece, pcm.size() - i);
            parakeet_capi_sound_stream_feed(s, pcm.data() + i, n, i + piece >= pcm.size(), &o, &no);
            for (int k = 0; k < no; ++k) {
                char b[128];
                std::snprintf(b, sizeof(b), "%d %.3f %.3f;", o[k].class_index, o[k].start, o[k].end);
                ref_sounds += b;
            }
            parakeet_capi_free_sound_segments(o);
        }
        parakeet_capi_sound_stream_free(s);
    }
    // 1c. the scene with all three
    parakeet_scene_opts o;
    parakeet_capi_scene_opts_default(&o);
    o.diar_latency = PARAKEET_DIAR_LATENCY_LOW;
    parakeet_scene_stream* sc = parakeet_capi_scene_stream_begin(asr, diar, tag, &o);
    CHECK(sc != nullptr, "scene begin");
    const std::string jl = feed_all_scene(sc, pcm, piece);
    parakeet_capi_scene_stream_free(sc);
    CHECK(utt_texts(jl) == ref, "scene utterances != sas_stream (%zu vs %zu)", utt_texts(jl).size(), ref.size());
    // Only the top-level "sounds" array (closed this call) counts here; the
    // "active" object has its own "sounds" array (still-open segments,
    // growing "end" on every call) that would otherwise match the same
    // "\"sounds\":[" needle and pollute the comparison.
    std::string got_sounds;
    {
        size_t line_start = 0;
        while (line_start < jl.size()) {
            size_t line_end = jl.find('\n', line_start);
            if (line_end == std::string::npos) line_end = jl.size();
            const size_t active_pos = jl.find("\"active\":", line_start);
            const size_t scan_end = (active_pos != std::string::npos && active_pos < line_end) ? active_pos : line_end;
            const size_t p = jl.find("\"sounds\":[", line_start);
            if (p != std::string::npos && p < scan_end) {
                size_t q = p, end = jl.find(']', p);
                while ((q = jl.find("{\"index\":", q)) != std::string::npos && q < end) {
                    int idx; float s0, s1;
                    std::sscanf(jl.c_str() + q, "{\"index\":%d,\"label\":\"%*[^\"]\",\"start\":%f,\"end\":%f", &idx, &s0, &s1);
                    char b[128];
                    std::snprintf(b, sizeof(b), "%d %.3f %.3f;", idx, s0, s1);
                    got_sounds += b;
                    ++q;
                }
            }
            line_start = line_end + 1;
        }
    }
    CHECK(got_sounds == ref_sounds, "scene sounds != sound_stream\n  %s\n  %s", got_sounds.c_str(), ref_sounds.c_str());
    CHECK(jl.find("Chicken, rooster") != std::string::npos, "rooster in scene");

    // 2. scene_sound_only
    parakeet_scene_stream* so = parakeet_capi_scene_stream_begin(nullptr, nullptr, tag, nullptr);
    CHECK(so != nullptr, "sound-only begin");
    const std::string jso = feed_all_scene(so, pcm, piece);
    CHECK(jso.find("\"words\":[{") == std::string::npos, "sound-only has words");
    CHECK(jso.find("\"sounds\":[{") != std::string::npos, "sound-only has no sounds");
    parakeet_capi_scene_stream_free(so);

    // 3. asr+tagger without diarization: speaker -1
    parakeet_scene_stream* at = parakeet_capi_scene_stream_begin(asr, nullptr, tag, nullptr);
    const std::string jat = feed_all_scene(at, pcm, piece);
    CHECK(jat.find("\"speaker\":-1") != std::string::npos, "no-diar words carry speaker -1");
    CHECK(jat.find("\"speaker\":0") == std::string::npos, "no-diar words carry a speaker");
    parakeet_capi_scene_stream_free(at);

    // 4. scene_wrong_kinds
    CHECK(parakeet_capi_scene_stream_begin(tag, nullptr, nullptr, nullptr) == nullptr, "tagger as ASR accepted");
    CHECK(std::strstr(parakeet_capi_last_error(tag), "CED sound model") != nullptr, "message: %s",
          parakeet_capi_last_error(tag));
    CHECK(parakeet_capi_scene_stream_begin(nullptr, nullptr, nullptr, nullptr) == nullptr, "no parts accepted");

    parakeet_capi_free(asr); parakeet_capi_free(diar); parakeet_capi_free(tag);
    if (fails) return 1;
    std::fprintf(stderr, "PASS\n");
    return 0;
}
