// SileroVad::load must refuse bad files with a message, not crash. No model needed.
#include <cstdio>
#include <fstream>
#include <string>

#include "ggml.h"
#include "gguf.h"
#include "silero_vad.hpp"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static const std::string kTmp = "test_silero_load_negative.tmp.gguf";

// load() must fail and the message must contain `needle`.
static void expect_fail(const std::string& path, const std::string& needle, int line) {
    std::string err;
    auto m = pk::SileroVad::load(path, &err);
    if (m || err.find(needle) == std::string::npos) {
        std::fprintf(stderr, "FAIL line %d: want an error with \"%s\", got %s \"%s\"\n", line, needle.c_str(),
                     m ? "success" : "failure", err.c_str());
        ++failures;
    }
}

int main() {
    expect_fail("does-not-exist.gguf", "cannot read", __LINE__);
    {
        std::ofstream f(kTmp, std::ios::binary);
        f << "this is not a GGUF file";
    }
    expect_fail(kTmp, "cannot read", __LINE__);
    {  // a GGUF without an architecture
        gguf_context* g = gguf_init_empty();
        gguf_set_val_str(g, "general.name", "x");
        CHECK(gguf_write_to_file(g, kTmp.c_str(), false));
        gguf_free(g);
    }
    expect_fail(kTmp, "general.architecture", __LINE__);
    {  // another architecture
        gguf_context* g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "parakeet");
        CHECK(gguf_write_to_file(g, kTmp.c_str(), false));
        gguf_free(g);
    }
    expect_fail(kTmp, "expected \"silero_vad\"", __LINE__);
    {  // right architecture, nothing else
        gguf_context* g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "silero_vad");
        CHECK(gguf_write_to_file(g, kTmp.c_str(), false));
        gguf_free(g);
    }
    expect_fail(kTmp, "silero_vad.lstm.hidden", __LINE__);
    {  // a key of the wrong type must not abort
        gguf_context* g = gguf_init_empty();
        gguf_set_val_str(g, "general.architecture", "silero_vad");
        gguf_set_val_str(g, "silero_vad.lstm.hidden", "128");
        CHECK(gguf_write_to_file(g, kTmp.c_str(), false));
        gguf_free(g);
    }
    expect_fail(kTmp, "silero_vad.lstm.hidden", __LINE__);
    std::remove(kTmp.c_str());

    if (failures) return 1;
    std::printf("test_silero_load_negative OK\n");
    return 0;
}
