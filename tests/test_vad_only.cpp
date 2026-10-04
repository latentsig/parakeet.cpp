// A VAD-only slice (parakeet.arch "vad", made by scripts/slice_vad_gguf.py):
// the C-API VAD calls give the same JSON as the parent model, and every other
// entry point refuses the file with a clear message. Plain ASR loading is
// unchanged: Model::load refuses the slice.
//
// LABEL model; run from the project root.
// Env: PARAKEET_TEST_VAD_ONLY_GGUF  slice of the parent below (skip 77 if unset)
//      PARAKEET_TEST_GGUF_ULTRA     the parent Ultra/Redux GGUF (skip 77 if unset)
#include "parakeet_capi.h"
#include "audio_io.hpp"
#include "model.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static std::string take(char* p) {
    if (!p) return std::string();
    std::string s(p);
    parakeet_capi_free_string(p);
    return s;
}

int main() {
    const char* slice = std::getenv("PARAKEET_TEST_VAD_ONLY_GGUF");
    const char* parent = std::getenv("PARAKEET_TEST_GGUF_ULTRA");
    if (!slice || !parent) { std::printf("SKIP: PARAKEET_TEST_VAD_ONLY_GGUF / PARAKEET_TEST_GGUF_ULTRA not set\n"); return 77; }

    // Three seconds of a tone burst between silences: enough to run the path.
    std::vector<float> pcm(16000 * 4, 0.0f);
    for (size_t i = 16000; i < 16000 * 3; ++i) pcm[i] = 0.3f * std::sin(0.05f * (float)i) * std::sin(0.0007f * (float)i);

    parakeet_ctx* s = parakeet_capi_load(slice);
    parakeet_ctx* f = parakeet_capi_load(parent);
    CHECK(s && f);
    if (!s || !f) return 1;

    const char* opts = "{\"probabilities\":true}";
    const std::string js = take(parakeet_capi_vad_pcm_json(s, pcm.data(), (int)pcm.size(), 16000, opts));
    const std::string jf = take(parakeet_capi_vad_pcm_json(f, pcm.data(), (int)pcm.size(), 16000, opts));
    CHECK(!js.empty());
    CHECK(js == jf);

    // Other entry points refuse the slice and name the reason.
    CHECK(parakeet_capi_transcribe_path(s, "x.wav", 0) == nullptr);
    const char* e = parakeet_capi_last_error(s);
    CHECK(e && std::strstr(e, "VAD-only"));
    CHECK(parakeet_capi_diarize_path(s, "x.wav") == nullptr);
    e = parakeet_capi_last_error(s);
    CHECK(e && std::strstr(e, "VAD-only"));

    // The ASR loader refuses the slice, and the VAD-only loader refuses a full model.
    CHECK(pk::Model::load(slice) == nullptr);
    CHECK(pk::Model::load_vad_only(parent) == nullptr);

    parakeet_capi_free(s);
    parakeet_capi_free(f);
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("OK\n");
    return 0;
}
