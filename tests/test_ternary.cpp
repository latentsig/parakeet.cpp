// Unit test for the ternary core: repack, activation quantization, scalar
// reference. Model-independent.
#include "ternary.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ggml.h"

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

// Upstream packing: 5 trits per byte, base 3, least significant digit first.
static std::vector<uint8_t> pack_trits(const std::vector<int>& codes, int N, int K) {
    const int nb = (K + 4) / 5;
    std::vector<uint8_t> q((size_t)N * nb, 0);
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < nb; ++b) {
            int v = 0, mul = 1;
            for (int d = 0; d < 5; ++d) {
                const int i = b * 5 + d;
                const int c = i < K ? codes[(size_t)n * K + i] : 0;
                v += c * mul;
                mul *= 3;
            }
            q[(size_t)n * nb + b] = (uint8_t)v;
        }
    return q;
}

struct Case { int N, K, T; };

static void run_case(const Case& cs, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(0, 2);
    std::uniform_real_distribution<float> sc(0.01f, 0.5f), xr(-3.0f, 3.0f);
    const int N = cs.N, K = cs.K, T = cs.T, G = K / kTernaryGroup;

    std::vector<int> codes((size_t)N * K);
    for (auto& c : codes) c = code(rng);
    std::vector<uint16_t> scales_f16((size_t)N * G);
    std::vector<float> scales_f((size_t)N * G);
    for (size_t i = 0; i < scales_f16.size(); ++i) {
        scales_f16[i] = ggml_fp32_to_fp16(sc(rng));
        scales_f[i] = ggml_fp16_to_fp32(scales_f16[i]);
    }
    const std::vector<uint8_t> q = pack_trits(codes, N, K);

    // dequant equals the definition
    std::vector<float> W;
    ternary_dequant(q.data(), scales_f16.data(), N, K, W);
    CHECK(W.size() == (size_t)N * K);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            CHECK(W[(size_t)n * K + k] == scales_f[(size_t)n * G + k / kTernaryGroup] * (float)(codes[(size_t)n * K + k] - 1));

    TernaryWeight w;
    ternary_repack(q.data(), scales_f16.data(), N, K, w);
    CHECK(w.N == N && w.K == K && w.groups() == G);
    CHECK(w.planes.size() == (size_t)N * G * 32);

    std::vector<float> x((size_t)T * K);
    for (auto& v : x) v = xr(rng);
    std::vector<uint8_t> act(ternary_act_row_bytes(K) * T);
    ternary_quant_rows(x.data(), K, 0, T, act.data());

    // ref matmul over the full row range, then over two halves: identical
    std::vector<float> y((size_t)T * N), y2((size_t)T * N, -1.0f);
    ternary_matmul_rows_ref(w, act.data(), T, y.data(), 0, N);
    const int mid = N / 2;
    ternary_matmul_rows_ref(w, act.data(), T, y2.data(), 0, mid);
    ternary_matmul_rows_ref(w, act.data(), T, y2.data(), mid, N);
    for (size_t i = 0; i < y.size(); ++i) CHECK(y[i] == y2[i]);

    // error against the float product: each activation is off by at most
    // scale/2, so |err| <= scale/2 * sum_k |W| (+ float rounding slack)
    for (int t = 0; t < T; ++t) {
        const float sa = *reinterpret_cast<const float*>(&act[(size_t)t * ternary_act_row_bytes(K) + K]);
        for (int n = 0; n < N; ++n) {
            double ref = 0.0, l1 = 0.0;
            for (int k = 0; k < K; ++k) {
                const double wv = W[(size_t)n * K + k];
                ref += wv * x[(size_t)t * K + k];
                l1 += std::fabs(wv);
            }
            const double bound = 0.5 * sa * l1 + 1e-4 * (std::fabs(ref) + 1.0);
            CHECK(std::fabs((double)y[(size_t)t * N + n] - ref) <= bound);
        }
    }
    // the dispatching entry point equals the reference exactly
    std::vector<float> yd((size_t)T * N, -1.0f);
    ternary_matmul_rows(w, act.data(), T, yd.data(), 0, N);
    for (size_t i = 0; i < y.size(); ++i) CHECK(y[i] == yd[i]);
}

static void test_silence_is_zero() {
    const int N = 4, K = 256, T = 2;
    std::vector<int> codes((size_t)N * K, 2);
    std::vector<uint16_t> s((size_t)N * (K / 128), ggml_fp32_to_fp16(1.0f));
    auto q = pack_trits(codes, N, K);
    TernaryWeight w;
    ternary_repack(q.data(), s.data(), N, K, w);
    std::vector<float> x((size_t)T * K, 0.0f);
    std::vector<uint8_t> act(ternary_act_row_bytes(K) * T);
    ternary_quant_rows(x.data(), K, 0, T, act.data());
    std::vector<float> y((size_t)T * N, 7.0f);
    ternary_matmul_rows(w, act.data(), T, y.data(), 0, N);
    for (float v : y) CHECK(v == 0.0f && !std::isnan(v));
}

int main() {
    const Case cases[] = {{8, 128, 1}, {16, 256, 3}, {37, 1024, 5}, {64, 4096, 4}, {5, 384, 7}, {1, 128, 1}};
    unsigned seed = 1;
    for (const Case& c : cases) run_case(c, seed++);
    test_silence_is_zero();
    if (failures) return 1;
    std::printf("test_ternary: OK (kernel=%s)\n", ternary_kernel_name());
    return 0;
}
