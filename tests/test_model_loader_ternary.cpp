// Loader reads parakeet.ternary.* / parakeet.vad.* and leaves other models alone.
// Skips (77) unless the env vars point at converted GGUFs.
#include "model_loader.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "ggml.h"

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

int main() {
    const char* keep = std::getenv("PARAKEET_TEST_GGUF_REDUX_KEEP");
    const char* plain = std::getenv("PARAKEET_TEST_GGUF");
    if (!keep && !plain) {
        std::puts("skip: PARAKEET_TEST_GGUF_REDUX_KEEP and PARAKEET_TEST_GGUF unset");
        return 77;
    }
    if (keep) {
        ModelLoader ml;
        CHECK(ml.load(keep));
        const ParakeetConfig& c = ml.config();
        CHECK(c.ternary.present);
        CHECK(c.ternary.group_size == 128);
        CHECK(c.vad.present);
        CHECK(c.vad.d_in == 1024 && c.vad.hidden == 128 && c.vad.kernel == 5);
        CHECK(std::fabs(c.vad.frame_sec - 0.08f) < 1e-6f);
        const ggml_tensor* q = ml.tensor("encoder.layers.0.self_attn.linear_q.qweight");
        CHECK(q && q->type == GGML_TYPE_I8 && q->ne[0] == 205 && q->ne[1] == 1024);
        const ggml_tensor* s = ml.tensor("encoder.layers.0.self_attn.linear_q.scales");
        CHECK(s && s->type == GGML_TYPE_F16 && s->ne[0] == 8 && s->ne[1] == 1024);
        CHECK(ml.tensor("encoder.layers.0.self_attn.linear_q.weight") == nullptr);
        const ggml_tensor* p = ml.tensor("encoder.layers.0.feed_forward1.linear2.qweight");
        CHECK(p && p->ne[0] == 820 && p->ne[1] == 1024);  // K = 4096, ceil(4096/5) = 820
        CHECK(ml.tensor("vad_head.proj.weight") != nullptr);
        CHECK(ml.tensor("vad_head.ctx.weight") != nullptr);
        CHECK(ml.tensor("vad_head.out.bias") != nullptr);
    }
    if (plain) {
        ModelLoader ml;
        CHECK(ml.load(plain));
        CHECK(!ml.config().ternary.present);
        CHECK(!ml.config().vad.present);
    }
    if (failures) return 1;
    std::puts("test_model_loader_ternary: OK");
    return 0;
}
