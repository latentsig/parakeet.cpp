// A doctored packed-ternary GGUF must be refused at load (Model::load returns
// nullptr) while the pristine file loads. Doctored copies are derived from
// $PARAKEET_TEST_GGUF_REDUX_KEEP with the gguf C API and written to the system
// temp directory. Skips (77) when the env var is unset.
#include "model.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>

#include "ggml.h"
#include "gguf.h"

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

// Loads src, applies edit, writes a temp file, returns its path ("" on error).
static std::string doctor(const char* src, const std::string& tag,
                          const std::function<void(gguf_context*, ggml_context*)>& edit) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{/*no_alloc*/ false, &ctx};
    gguf_context* g = gguf_init_from_file(src, p);
    if (!g) return "";
    // Rebuild a fresh gguf (the loaded one carries stale tensor infos that the
    // writer cannot use): copy the KVs, then add every tensor with its data.
    gguf_context* w = gguf_init_empty();
    gguf_set_kv(w, g);
    const int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n; ++i)
        gguf_add_tensor(w, ggml_get_tensor(ctx, gguf_get_tensor_name(g, i)));
    edit(w, ctx);
    const std::string out = (std::filesystem::temp_directory_path() / ("pk_neg_" + tag + ".gguf")).string();
    const bool ok = gguf_write_to_file(w, out.c_str(), false);
    gguf_free(w);
    gguf_free(g);
    ggml_free(ctx);
    return ok ? out : "";
}

static void expect_refused(const char* src, const std::string& tag,
                           const std::function<void(gguf_context*, ggml_context*)>& edit) {
    const std::string path = doctor(src, tag, edit);
    CHECK(!path.empty());
    if (path.empty()) return;
    auto m = Model::load(path);
    std::printf("%s: %s\n", tag.c_str(), m ? "LOADED (unexpected)" : "refused");
    CHECK(m == nullptr);
    std::remove(path.c_str());
}

int main() {
    const char* keep = std::getenv("PARAKEET_TEST_GGUF_REDUX_KEEP");
    if (!keep) {
        std::puts("skip: PARAKEET_TEST_GGUF_REDUX_KEEP unset");
        return 77;
    }
    {
        auto m = Model::load(keep);
        CHECK(m != nullptr);  // pristine file loads
    }
    expect_refused(keep, "flag_off", [](gguf_context* g, ggml_context*) {
        gguf_set_val_bool(g, "parakeet.ternary.present", false);
    });
    expect_refused(keep, "group64", [](gguf_context* g, ggml_context*) {
        gguf_set_val_u32(g, "parakeet.ternary.group_size", 64);
    });
    expect_refused(keep, "scales_type", [](gguf_context* g, ggml_context*) {
        // I8 is half the byte size of F16, so the writer stays inside the buffer.
        gguf_set_tensor_type(g, "encoder.layers.0.feed_forward1.linear1.scales", GGML_TYPE_I8);
    });
    expect_refused(keep, "ff_dim", [](gguf_context* g, ggml_context*) {
        gguf_set_val_u32(g, "parakeet.encoder.ff_dim", 2048);
    });
    expect_refused(keep, "vad_nan_frame", [](gguf_context* g, ggml_context*) {
        gguf_set_val_f32(g, "parakeet.vad.frame_sec", NAN);
    });
    expect_refused(keep, "vad_even_kernel", [](gguf_context* g, ggml_context*) {
        gguf_set_val_u32(g, "parakeet.vad.kernel", 4);
    });
    if (failures) return 1;
    std::puts("test_ternary_load_negative: OK");
    return 0;
}
