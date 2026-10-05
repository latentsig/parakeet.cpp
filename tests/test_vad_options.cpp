// VAD options and the C-API VAD entry points that need no model: the
// per-kind defaults, "speech_pad", error messages, and the NULL / bad-file
// paths of the context and stream functions.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "ggml.h"
#include "gguf.h"
#include "parakeet_capi.h"
#include "vad_json.hpp"

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)
static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

static bool parse(const char* json, VadKind kind, VadRequest& r, std::string& err) {
    err.clear();
    return parse_vad_options(json, r, err, kind);
}

int main() {
    VadRequest r;
    std::string err;

    // Defaults of the head are the segmenter defaults (unchanged by Silero).
    CHECK(parse(nullptr, VadKind::kHead, r, err));
    CHECK(r.kind == VadKind::kHead && r.mode == VadRequest::Mode::kSpeech && !r.probabilities);
    CHECK(r.opts.threshold == 0.5f && near(r.opts.min_pause_sec, 0.2) && near(r.opts.min_speech_sec, 0.1) &&
          near(r.opts.max_seg_sec, 30.0) && near(r.opts.bridge_sec, 0.1) && r.opts.pad_sec == 0.0 &&
          near(r.opts.frame_sec, 0.08));
    CHECK(parse("", VadKind::kHead, r, err) && near(r.opts.min_pause_sec, 0.2));
    CHECK(parse(" {} ", VadKind::kHead, r, err) && near(r.opts.min_pause_sec, 0.2));

    // Silero: 250 ms / 100 ms / 30 ms, threshold 0.5, 32 ms frames.
    CHECK(parse(nullptr, VadKind::kSilero, r, err));
    CHECK(r.kind == VadKind::kSilero && r.opts.threshold == 0.5f && near(r.opts.min_speech_sec, 0.25) &&
          near(r.opts.min_pause_sec, 0.1) && near(r.opts.pad_sec, 0.03) && near(r.opts.frame_sec, 0.032) &&
          near(r.opts.max_seg_sec, 30.0));

    // Keys override the kind's defaults and leave the others alone.
    CHECK(parse("{\"min_speech\":0.5,\"speech_pad\":0,\"threshold\":0.7,\"mode\":\"segments\","
                "\"probabilities\":true,\"max_segment\":12,\"min_pause\":0.3}", VadKind::kSilero, r, err));
    CHECK(near(r.opts.min_speech_sec, 0.5) && r.opts.pad_sec == 0.0 && std::fabs(r.opts.threshold - 0.7f) < 1e-6 &&
          r.mode == VadRequest::Mode::kSegments && r.probabilities && near(r.opts.max_seg_sec, 12) &&
          near(r.opts.min_pause_sec, 0.3) && near(r.opts.bridge_sec, 0.1));
    CHECK(parse("{\"speech_pad\":0.05}", VadKind::kHead, r, err) && near(r.opts.pad_sec, 0.05));

    // Errors name the option.
    const struct { const char* json; const char* word; } bad[] = {
        {"{\"speech_pad\":-0.1}", "speech_pad"}, {"{\"speech_pad\":\"x\"}", "speech_pad"},
        {"{\"threshold\":0}", "threshold"}, {"{\"threshold\":1.01}", "threshold"},
        {"{\"min_speech\":0}", "min_speech"}, {"{\"min_pause\":-1}", "min_pause"},
        {"{\"mode\":\"stream\"}", "mode"}, {"{\"nope\":1}", "nope"}, {"[1]", "JSON object"},
        {"{\"threshold\":", "threshold"}, {"{} x", "trailing"}, {"{\"probabilities\":1}", "probabilities"},
    };
    for (const auto& b : bad) {
        for (VadKind k : {VadKind::kHead, VadKind::kSilero}) {
            CHECK(!parse(b.json, k, r, err));
            CHECK(err.find(b.word) != std::string::npos);
        }
    }

    // "trim": default 0.3 for both kinds, 0 turns it off, negative is an error.
    CHECK(parse(nullptr, VadKind::kHead, r, err) && near(r.opts.trim_sec, 0.3));
    CHECK(parse(nullptr, VadKind::kSilero, r, err) && near(r.opts.trim_sec, 0.3));
    CHECK(parse("{\"trim\":0}", VadKind::kHead, r, err) && r.opts.trim_sec == 0.0 && near(r.opts.max_seg_sec, 30.0));
    CHECK(parse("{\"trim\":0.5,\"mode\":\"segments\"}", VadKind::kSilero, r, err) && near(r.opts.trim_sec, 0.5) &&
          r.mode == VadRequest::Mode::kSegments && near(r.opts.pad_sec, 0.03));
    for (VadKind k : {VadKind::kHead, VadKind::kSilero}) {
        CHECK(!parse("{\"trim\":-0.1}", k, r, err) && err.find("trim") != std::string::npos);
        CHECK(!parse("{\"trim\":\"x\"}", k, r, err) && err.find("trim") != std::string::npos);
        CHECK(!parse("{\"trim\":1e9}", k, r, err) && err.find("trim") != std::string::npos);
    }

    // Word filter keys: only with allow_filter; off by default.
    CHECK(parse(nullptr, VadKind::kHead, r, err) && !r.filter.active() && near(r.filter.local_radius_sec, 5.0));
    CHECK(!parse_vad_options("{\"min_local_conf\":0.5}", r, err, VadKind::kHead, false) &&
          err.find("min_local_conf") != std::string::npos);
    CHECK(!parse_vad_options("{\"drop_punct_only\":true}", r, err, VadKind::kHead, false) &&
          err.find("drop_punct_only") != std::string::npos);
    CHECK(parse_vad_options("{\"min_local_conf\":0.5,\"local_radius\":3,\"drop_punct_only\":true,\"trim\":0.2,\"max_segment\":10}",
                            r, err, VadKind::kSilero, true));
    CHECK(r.filter.min_local_conf == 0.5f && r.filter.local_radius_sec == 3.0f && r.filter.drop_punct_only &&
          r.filter.active() && near(r.opts.trim_sec, 0.2) && near(r.opts.max_seg_sec, 10.0));
    CHECK(parse_vad_options("{\"min_local_conf\":0,\"drop_punct_only\":false}", r, err, VadKind::kHead, true) && !r.filter.active());
    CHECK(parse_vad_options("{\"min_local_conf\":1}", r, err, VadKind::kHead, true) && r.filter.min_local_conf == 1.0f);
    {
        const struct { const char* json; const char* word; } badf[] = {
            {"{\"min_local_conf\":1.01}", "min_local_conf"}, {"{\"min_local_conf\":-0.5}", "min_local_conf"},
            {"{\"min_local_conf\":\"x\"}", "min_local_conf"}, {"{\"local_radius\":0}", "local_radius"},
            {"{\"local_radius\":-1}", "local_radius"}, {"{\"local_radius\":1e9}", "local_radius"},
            {"{\"drop_punct_only\":1}", "drop_punct_only"}, {"{\"drop_punct_only\":\"true\"}", "drop_punct_only"},
        };
        for (const auto& b : badf) {
            CHECK(!parse_vad_options(b.json, r, err, VadKind::kHead, true));
            CHECK(err.find(b.word) != std::string::npos);
        }
    }
    // parse_filter_options: the filter keys alone.
    {
        WordFilter f;
        CHECK(parse_filter_options(nullptr, f, err) && !f.active());
        CHECK(parse_filter_options("", f, err) && !f.active());
        CHECK(parse_filter_options("{}", f, err) && !f.active() && f.local_radius_sec == 5.0f);
        CHECK(parse_filter_options("{\"min_local_conf\":0.5}", f, err) && f.min_local_conf == 0.5f && f.local_radius_sec == 5.0f &&
              !f.drop_punct_only);
        CHECK(parse_filter_options("{\"drop_punct_only\":true,\"local_radius\":2.5}", f, err) && f.drop_punct_only &&
              f.min_local_conf == 0.0f && f.local_radius_sec == 2.5f);
        const char* bad_only[] = {"{\"threshold\":0.5}", "{\"trim\":0.3}", "{\"mode\":\"speech\"}", "{\"max_segment\":5}",
                                  "{\"nope\":1}", "[1]", "{\"min_local_conf\":", "{} x", "{\"min_local_conf\":2}"};
        for (const char* j : bad_only) CHECK(!parse_filter_options(j, f, err) && !err.empty());
        CHECK(!parse_filter_options("{\"threshold\":0.5}", f, err) && err.find("threshold") != std::string::npos &&
              err.find("unknown") != std::string::npos);
    }

    // C-API: NULL and wrong-kind contexts fail without a crash.
    CHECK(parakeet_capi_vad_pcm_json(nullptr, nullptr, 0, 16000, nullptr) == nullptr);
    CHECK(parakeet_capi_vad_path_json(nullptr, "x.wav", nullptr) == nullptr);
    CHECK(parakeet_capi_vad_stream_begin(nullptr, 16000, nullptr) == nullptr);
    CHECK(parakeet_capi_vad_stream_feed_json(nullptr, nullptr, 0, 0) == nullptr);
    CHECK(parakeet_capi_vad_stream_reset(nullptr) != 0);
    parakeet_capi_vad_stream_free(nullptr);
    CHECK(parakeet_capi_transcribe_path_json_vad_with(nullptr, nullptr, "x.wav", 0, nullptr) == nullptr);
    CHECK(parakeet_capi_transcribe_path_json_with(nullptr, "x.wav", 0, nullptr) == nullptr);
    CHECK(parakeet_capi_model_kind(nullptr) == PARAKEET_MODEL_KIND_NONE);
    CHECK(parakeet_capi_load("does-not-exist.gguf") == nullptr);

    // A GGUF that claims to be Silero but is empty must fail to load, not crash.
    const char* tmp = "test_vad_options.tmp.gguf";
    {
        gguf_context* g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "silero_vad");
        CHECK(gguf_write_to_file(g, tmp, false));
        gguf_free(g);
    }
    CHECK(gguf_is_silero(tmp));
    CHECK(parakeet_capi_load(tmp) == nullptr);
    {
        gguf_context* g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "parakeet");
        CHECK(gguf_write_to_file(g, tmp, false));
        gguf_free(g);
    }
    CHECK(!gguf_is_silero(tmp));
    std::remove(tmp);
    CHECK(!gguf_is_silero("does-not-exist.gguf"));

    if (failures) return 1;
    std::puts("test_vad_options: OK");
    return 0;
}
