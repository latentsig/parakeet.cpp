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

// d_in = 2, hidden = 2, kernel = 3, deliberately asymmetric so a transposed
// [in][out] indexing bug cannot pass.
//   proj: h0 = x0 + 2*x1 + 0.5, h1 = x1 - 0.5
//   ctx (taps kk = 0, 1, 2 are t-1, t, t+1):
//     o0 = 1 * in1(t) + 2 * in0(t+1) + 0.25      (o != i, and a right-hand tap)
//     o1 = 1 * in0(t-1) - 0.5                    (o != i, left tap)
//   out: z = 0.1 * o0 + 0.05 * o1 - 0.5
static VadWeights tiny() {
    VadWeights w;
    w.d_in = 2; w.hidden = 2; w.kernel = 3;
    w.proj_w = {1, 2, 0, 1};
    w.proj_b = {0.5f, -0.5f};
    w.ctx_w.assign(2 * 2 * 3, 0.0f);
    w.ctx_w[(0 * 2 + 1) * 3 + 1] = 1.0f;   // o=0, i=1, kk=1 (center)
    w.ctx_w[(0 * 2 + 0) * 3 + 2] = 2.0f;   // o=0, i=0, kk=2 (t+1)
    w.ctx_w[(1 * 2 + 0) * 3 + 0] = 1.0f;   // o=1, i=0, kk=0 (t-1)
    w.ctx_b = {0.25f, -0.5f};
    w.out_w = {0.1f, 0.05f};
    w.out_b = -0.5f;
    return w;
}

// x = (1,0), (0,1), (2,1). After proj, before ReLU:
//   t0: (1.5, -0.5)   t1: (2.5, 0.5)   t2: (4.5, 0.5)
// after ReLU h1: t0 (1.5, 0), t1 (2.5, 0.5), t2 (4.5, 0.5).
// ctx pre-activation (right pad and left pad are zero):
//   t0: o0 = h1[0][1] + 2*h1[1][0] + 0.25 = 0 + 5 + 0.25 = 5.25
//       o1 = (left pad 0) - 0.5 = -0.5
//   t1: o0 = 0.5 + 2*4.5 + 0.25 = 9.75      o1 = h1[0][0] - 0.5 = 1.0
//   t2: o0 = 0.5 + (right pad 0) + 0.25 = 0.75   o1 = h1[1][0] - 0.5 = 2.0
static const float kX[] = {1, 0,   0, 1,   2, 1};

static void test_plain_variant() {
    VadVariant v;  // relu after proj, no residual, relu after ctx
    v.relu_after_proj = true; v.residual = false; v.relu_after_ctx = true;
    const auto p = VadHead::run(tiny(), v, kX, 3);
    CHECK(p.size() == 3);
    // ReLU after ctx: t0 (5.25, 0), t1 (9.75, 1.0), t2 (0.75, 2.0)
    // z0 = 0.1*5.25 + 0.05*0    - 0.5 = 0.025
    // z1 = 0.1*9.75 + 0.05*1.0  - 0.5 = 0.525
    // z2 = 0.1*0.75 + 0.05*2.0  - 0.5 = -0.325
    CHECK(near(p[0], sig(0.025f)));
    CHECK(near(p[1], sig(0.525f)));
    CHECK(near(p[2], sig(-0.325f)));
}

static void test_residual_variant() {
    VadVariant v;
    v.relu_after_proj = true; v.residual = true; v.relu_after_ctx = true;
    const auto p = VadHead::run(tiny(), v, kX, 3);
    // Residual adds h1 (post-ReLU) before the ReLU after ctx:
    //   t0: (5.25+1.5, -0.5+0)   = (6.75, -0.5 -> 0)
    //   t1: (9.75+2.5, 1.0+0.5)  = (12.25, 1.5)
    //   t2: (0.75+4.5, 2.0+0.5)  = (5.25, 2.5)
    // z0 = 0.675 - 0.5 = 0.175
    // z1 = 1.225 + 0.075 - 0.5 = 0.8
    // z2 = 0.525 + 0.125 - 0.5 = 0.15
    CHECK(near(p[0], sig(0.175f)));
    CHECK(near(p[1], sig(0.8f)));
    CHECK(near(p[2], sig(0.15f)));
}

static void test_relu_matters() {
    const float x[] = {-1, -2};  // T = 1, negative inputs
    VadVariant with, without;
    with.relu_after_proj = true;  with.relu_after_ctx = true;
    without.relu_after_proj = false; without.relu_after_ctx = false;
    const auto a = VadHead::run(tiny(), with, x, 1);
    const auto b = VadHead::run(tiny(), without, x, 1);
    // with ReLUs: proj pre (-1-4+0.5, -2-0.5) = (-4.5, -2.5) -> h1 = (0, 0);
    //   ctx (0.25, -0.5) -> ReLU (0.25, 0); z = 0.025 - 0.5 = -0.475
    CHECK(near(a[0], sig(-0.475f)));
    // without: h1 = (-4.5, -2.5). T = 1 so only the center tap is in range:
    //   o0 = in1 + 0.25 = -2.5 + 0.25 = -2.25, o1 = -0.5 (its tap is padding)
    //   z = 0.1*-2.25 + 0.05*-0.5 - 0.5 = -0.225 - 0.025 - 0.5 = -0.75
    CHECK(near(b[0], sig(-0.75f)));
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
