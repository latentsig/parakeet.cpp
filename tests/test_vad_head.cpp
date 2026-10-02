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

// x = (1,0), (0,1), (2,1). After proj, before the activation:
//   t0: (1.5, -0.5)   t1: (2.5, 0.5)   t2: (4.5, 0.5)
// The ctx pre-activation uses h1 = act(proj) (right pad and left pad are zero):
//   t0: o0 = h1[0][1] + 2*h1[1][0] + 0.25      o1 = -0.5
//   t1: o0 = h1[1][1] + 2*h1[2][0] + 0.25      o1 = h1[0][0] - 0.5
//   t2: o0 = h1[2][1] + 0.25                   o1 = h1[1][0] - 0.5
static const float kX[] = {1, 0,   0, 1,   2, 1};
static float silu(float z) { return z * sig(z); }
static float relu(float z) { return z > 0.0f ? z : 0.0f; }

// Expected probabilities from the hand-derived pre-activations above, for any
// activation functions and with or without the residual.
static void expect_tiny(const std::vector<float>& p, float (*a1)(float), float (*a2)(float), bool residual) {
    const float h[3][2] = {{a1(1.5f), a1(-0.5f)}, {a1(2.5f), a1(0.5f)}, {a1(4.5f), a1(0.5f)}};
    float o[3][2] = {{h[0][1] + 2 * h[1][0] + 0.25f, -0.5f},
                     {h[1][1] + 2 * h[2][0] + 0.25f, h[0][0] - 0.5f},
                     {h[2][1] + 0.25f, h[1][0] - 0.5f}};
    CHECK(p.size() == 3);
    if (p.size() != 3) return;
    for (int t = 0; t < 3; ++t) {
        for (int k = 0; k < 2; ++k) o[t][k] = a2(o[t][k] + (residual ? h[t][k] : 0.0f));
        CHECK(near(p[(size_t)t], sig(0.1f * o[t][0] + 0.05f * o[t][1] - 0.5f)));
    }
}

static void test_default_is_silu_no_residual() {
    const VadVariant v;
    CHECK(v.act_proj == VadAct::kSiLU && v.act_ctx == VadAct::kSiLU && !v.residual);
    expect_tiny(VadHead::run(tiny(), v, kX, 3), silu, silu, false);
}

static void test_relu_variant() {
    VadVariant v;
    v.act_proj = VadAct::kReLU; v.act_ctx = VadAct::kReLU;
    expect_tiny(VadHead::run(tiny(), v, kX, 3), relu, relu, false);
}

static void test_residual_variant() {
    VadVariant v;
    v.residual = true;
    expect_tiny(VadHead::run(tiny(), v, kX, 3), silu, silu, true);
}

static void test_activation_matters() {
    const float x[] = {-1, -2};  // T = 1, negative inputs
    VadVariant silu_v, relu_v;
    relu_v.act_proj = VadAct::kReLU; relu_v.act_ctx = VadAct::kReLU;
    const auto a = VadHead::run(tiny(), silu_v, x, 1);
    const auto b = VadHead::run(tiny(), relu_v, x, 1);
    // proj pre (-4.5, -2.5). T = 1, so only the center tap is in range.
    // SiLU: h1 = (silu(-4.5), silu(-2.5)); o0 = h1[1] + 0.25, o1 = -0.5.
    const float h1 = silu(-2.5f);
    const float o0 = silu(h1 + 0.25f), o1 = silu(-0.5f);
    CHECK(near(a[0], sig(0.1f * o0 + 0.05f * o1 - 0.5f)));
    // ReLU: h1 = (0, 0); ctx (0.25, -0.5) -> (0.25, 0); z = 0.025 - 0.5.
    CHECK(near(b[0], sig(-0.475f)));
    CHECK(std::fabs(a[0] - b[0]) > 1e-3f);
}

static void test_from_index() {
    // Index 0 is the default wiring; each bit swaps one choice.
    const VadVariant d = VadVariant::from_index(0);
    CHECK(d.act_proj == VadAct::kSiLU && d.act_ctx == VadAct::kSiLU && !d.residual);
    const VadVariant v = VadVariant::from_index(5);
    CHECK(v.act_proj == VadAct::kReLU && !v.residual && v.act_ctx == VadAct::kReLU);
    const VadVariant r = VadVariant::from_index(2);
    CHECK(r.act_proj == VadAct::kSiLU && r.residual && r.act_ctx == VadAct::kSiLU);
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
    test_default_is_silu_no_residual();
    test_relu_variant();
    test_residual_variant();
    test_activation_matters();
    test_from_index();
    test_real_model();
    if (failures) return 1;
    std::puts("test_vad_head: OK");
    return 0;
}
