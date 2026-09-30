// VadHead math on hand-checkable weights, plus a smoke test on a real model.
#include "vad_head.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "model_loader.hpp"

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)
static bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }
static float sig(float z) { return 1.0f / (1.0f + std::exp(-z)); }

// d_in = 2, hidden = 2, kernel = 3. proj = identity. ctx: out0 = center tap of
// in0, out1 = left tap (t-1) of in1. out = [1, 1], bias -1.
static VadWeights tiny() {
    VadWeights w;
    w.d_in = 2; w.hidden = 2; w.kernel = 3;
    w.proj_w = {1, 0, 0, 1};
    w.proj_b = {0, 0};
    w.ctx_w.assign(2 * 2 * 3, 0.0f);
    w.ctx_w[(0 * 2 + 0) * 3 + 1] = 1.0f;   // o=0, i=0, kk=1 (center)
    w.ctx_w[(1 * 2 + 1) * 3 + 0] = 1.0f;   // o=1, i=1, kk=0 (t-1)
    w.ctx_b = {0, 0};
    w.out_w = {1, 1};
    w.out_b = -1.0f;
    return w;
}

static void test_plain_variant() {
    const float x[] = {1, 2,   0, 3,   2, 0};  // T = 3
    VadVariant v;  // relu after proj, no residual, relu after ctx
    v.relu_after_proj = true; v.residual = false; v.relu_after_ctx = true;
    const auto p = VadHead::run(tiny(), v, x, 3);
    CHECK(p.size() == 3);
    CHECK(near(p[0], sig(0.0f)));
    CHECK(near(p[1], sig(1.0f)));
    CHECK(near(p[2], sig(4.0f)));
}

static void test_residual_variant() {
    const float x[] = {1, 2,   0, 3,   2, 0};
    VadVariant v;
    v.relu_after_proj = true; v.residual = true; v.relu_after_ctx = true;
    const auto p = VadHead::run(tiny(), v, x, 3);
    CHECK(near(p[0], sig(3.0f)));
    CHECK(near(p[1], sig(4.0f)));
    CHECK(near(p[2], sig(6.0f)));
}

static void test_relu_matters() {
    const float x[] = {-1, -2};  // T = 1, negative inputs
    VadVariant with, without;
    with.relu_after_proj = true;  with.relu_after_ctx = true;
    without.relu_after_proj = false; without.relu_after_ctx = false;
    const auto a = VadHead::run(tiny(), with, x, 1);
    const auto b = VadHead::run(tiny(), without, x, 1);
    CHECK(near(a[0], sig(-1.0f)));              // everything clipped to 0, bias only
    CHECK(near(b[0], sig(-1.0f + -1.0f)));      // center tap of in0 = -1, left tap of in1 = 0 (pad)
}

static void test_from_index() {
    const VadVariant v = VadVariant::from_index(5);
    CHECK(v.relu_after_proj && !v.residual && v.relu_after_ctx);
    const VadVariant z = VadVariant::from_index(0);
    CHECK(!z.relu_after_proj && !z.residual && !z.relu_after_ctx);
}

static void test_real_model() {
    const char* p = std::getenv("PARAKEET_TEST_GGUF_ULTRA");
    if (!p) return;
    ModelLoader ml;
    CHECK(ml.load(p));
    VadHead head(ml);
    CHECK(std::fabs(head.frame_sec() - 0.08f) < 1e-6f);
    const int T = 40, D = (int)ml.config().vad.d_in;
    std::vector<float> x((size_t)T * D);
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.01f * (float)i);
    const auto prob = head.probabilities(x.data(), T);
    CHECK((int)prob.size() == T);
    for (float v : prob) CHECK(std::isfinite(v) && v >= 0.0f && v <= 1.0f);
}

int main() {
    test_plain_variant();
    test_residual_variant();
    test_relu_matters();
    test_from_index();
    test_real_model();
    if (failures) return 1;
    std::puts("test_vad_head: OK");
    return 0;
}
