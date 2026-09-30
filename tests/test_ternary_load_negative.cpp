// A doctored packed-ternary GGUF must be refused at load (Model::load returns
// nullptr) while the pristine file loads. Doctored copies are derived from
// $PARAKEET_TEST_GGUF_REDUX_KEEP with the gguf C API and written to the system
// temp directory. Skips (77) when the env var is unset.
#include "model.hpp"
#include "model_loader.hpp"
#include "ternary.hpp"

#include <unistd.h>
#include <cstring>

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

// Removes the file on every exit path.
struct TempFile {
    std::string path;
    explicit TempFile(const std::string& tag) {
        path = (std::filesystem::temp_directory_path() /
                ("pk_neg_" + std::to_string((long)getpid()) + "_" + tag + ".gguf")).string();
    }
    ~TempFile() { std::remove(path.c_str()); }
};

// Rebuilds src through gguf_init_empty (the writer cannot reuse the loaded
// context's tensor infos), skipping tensors for which drop(name) is true, then
// applies edit and writes to tf.path. Returns false on error.
static bool doctor(const char* src, const TempFile& tf,
                   const std::function<void(gguf_context*, ggml_context*)>& edit,
                   const std::function<bool(const std::string&)>& drop = nullptr) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{/*no_alloc*/ false, &ctx};
    gguf_context* g = gguf_init_from_file(src, p);
    if (!g) return false;
    gguf_context* w = gguf_init_empty();
    gguf_set_kv(w, g);
    const int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n; ++i) {
        const char* nm = gguf_get_tensor_name(g, i);
        if (drop && drop(nm)) continue;
        gguf_add_tensor(w, ggml_get_tensor(ctx, nm));
    }
    edit(w, ctx);
    const bool ok = gguf_write_to_file(w, tf.path.c_str(), false);
    gguf_free(w);
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

static void expect_refused(const char* src, const std::string& tag,
                           const std::function<void(gguf_context*, ggml_context*)>& edit) {
    TempFile tf(tag);
    const bool ok = doctor(src, tf, edit);
    CHECK(ok);
    if (!ok) return;
    auto m = Model::load(tf.path);
    std::printf("%s: %s\n", tag.c_str(), m ? "LOADED (unexpected)" : "refused");
    CHECK(m == nullptr);
}

static bool ends_with(const std::string& s, const char* suf) {
    const size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
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
    // Control: an unedited copy through the same rebuild path must load, so a
    // writer regression cannot make every case below pass vacuously.
    {
        TempFile tf("control");
        CHECK(doctor(keep, tf, [](gguf_context*, ggml_context*) {}));
        auto m = Model::load(tf.path);
        CHECK(m != nullptr);
        ModelLoader ml;
        CHECK(ml.load(tf.path));
        CHECK(ternary_flag_consistency_error(ml).empty());
    }
    // Flag off with every packed tensor still present.
    {
        TempFile tf("flag_off");
        CHECK(doctor(keep, tf, [](gguf_context* g, ggml_context*) {
            gguf_set_val_bool(g, "parakeet.ternary.present", false);
        }));
        ModelLoader ml;
        CHECK(ml.load(tf.path));
        const std::string err = ternary_flag_consistency_error(ml);
        CHECK(err.find("qweight") != std::string::npos);
        CHECK(Model::load(tf.path) == nullptr);
    }
    // Flag off, layer 0 has no packed tensors (dropped), later layers keep
    // theirs: the old two-tensor layer-0 probe missed this.
    {
        TempFile tf("flag_off_later_layer");
        CHECK(doctor(keep, tf, [](gguf_context* g, ggml_context*) {
            gguf_set_val_bool(g, "parakeet.ternary.present", false);
        }, [](const std::string& n) {
            return n.rfind("encoder.layers.0.", 0) == 0 && (ends_with(n, ".qweight") || ends_with(n, ".scales"));
        }));
        ModelLoader ml;
        CHECK(ml.load(tf.path));
        CHECK(!ml.tensor("encoder.layers.0.self_attn.linear_q.qweight"));
        CHECK(!ml.tensor("encoder.layers.0.feed_forward1.linear1.qweight"));
        CHECK(ml.has_tensor_with_suffix(".qweight"));
        const std::string err = ternary_flag_consistency_error(ml);
        std::printf("flag_off_later_layer: %s\n", err.c_str());
        CHECK(err.find("qweight") != std::string::npos);
        CHECK(Model::load(tf.path) == nullptr);
    }
    // Flag on but no packed tensors at all.
    {
        TempFile tf("flag_on_no_packed");
        CHECK(doctor(keep, tf, [](gguf_context*, ggml_context*) {},
                     [](const std::string& n) { return ends_with(n, ".qweight") || ends_with(n, ".scales"); }));
        ModelLoader ml;
        CHECK(ml.load(tf.path));
        CHECK(ternary_flag_consistency_error(ml).find("qweight") != std::string::npos);
        CHECK(Model::load(tf.path) == nullptr);
    }
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
