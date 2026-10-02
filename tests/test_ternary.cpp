// Unit test for the ternary core: repack, activation quantization, scalar
// reference. Model-independent.
#include "ternary.hpp"
#include "ternary_kernels.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
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

// wide: codes mostly 2 and positive activations, so group sums reach 14 bits
// and scale * sum no longer fits a float mantissa exactly. Only then does a
// fused multiply-add round differently from the reference's multiply and add.
struct Case { int N, K, T; bool wide = false; };

static void run_case(const Case& cs, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code(0, 2);
    std::uniform_real_distribution<float> sc(0.01f, 0.5f), xr(-3.0f, 3.0f);
    const int N = cs.N, K = cs.K, T = cs.T, G = K / kTernaryGroup;

    std::vector<int> codes((size_t)N * K);
    std::uniform_int_distribution<int> pct(0, 99);
    for (auto& c : codes) c = cs.wide ? (pct(rng) < 90 ? 2 : code(rng)) : code(rng);
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
    CHECK(w.planes.size() == (size_t)w.row_blocks() * G * 512);
    CHECK(w.scales.size() == (size_t)w.row_blocks() * G * kTernaryRowBlock);
    // the repacked layout holds exactly the input codes and scales
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) CHECK(w.code(n, k) == codes[(size_t)n * K + k]);
        for (int g = 0; g < G; ++g) CHECK(w.scale(n, g) == scales_f[(size_t)n * G + g]);
    }

    std::vector<float> x((size_t)T * K);
    std::uniform_real_distribution<float> xw(1.0f, 3.0f);
    for (auto& v : x) v = cs.wide ? xw(rng) : xr(rng);
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
    // tight check: scalar reference vs a double product of the dequantized
    // weights and the dequantized activations (catches a mis-indexed group scale)
    for (int t = 0; t < T; ++t) {
        const uint8_t* row = &act[(size_t)t * ternary_act_row_bytes(K)];
        const int8_t* qa = reinterpret_cast<const int8_t*>(row);
        const float sa = *reinterpret_cast<const float*>(row + K);
        for (int n = 0; n < N; ++n) {
            double ref = 0.0, mag = 0.0;
            for (int k = 0; k < K; ++k) {
                const double term = (double)W[(size_t)n * K + k] * ((double)qa[k] * (double)sa);
                ref += term;
                mag += std::fabs(term);
            }
            CHECK(std::fabs((double)y[(size_t)t * N + n] - ref) <= 1e-4 * (mag + 1e-6));
        }
    }
    // every available kernel equals the reference exactly
    for (const TernaryKernel* k : ternary_all_kernels()) {
        std::vector<float> yk((size_t)T * N, -1.0f);
        // one call over all rows, then a split with an uneven boundary
        k->fn(w, act.data(), T, yk.data(), 0, N);
        for (size_t i = 0; i < y.size(); ++i)
            if (y[i] != yk[i]) { std::fprintf(stderr, "kernel %s differs at %zu: %g vs %g (N=%d K=%d T=%d)\n", k->name, i, y[i], yk[i], N, K, T); ++failures; break; }
        std::vector<float> ys((size_t)T * N, -1.0f);
        const int b = N > 2 ? N / 3 : 1;
        k->fn(w, act.data(), T, ys.data(), 0, b);
        k->fn(w, act.data(), T, ys.data(), b, N);
        for (size_t i = 0; i < y.size(); ++i) CHECK(y[i] == ys[i]);
        // the op's split by thread count, including more threads than rows:
        // every output written exactly by its own range, nothing else touched
        for (int nth : {2, 3, 5, 7, 16, 40}) {
            std::vector<float> yt((size_t)T * N, -1.0f);
            for (int ith = 0; ith < nth; ++ith)
                k->fn(w, act.data(), T, yt.data(), (int)((int64_t)N * ith / nth), (int)((int64_t)N * (ith + 1) / nth));
            for (size_t i = 0; i < y.size(); ++i) CHECK(y[i] == yt[i]);
        }
        // a single inner range leaves everything outside it untouched
        if (N >= 3) {
            const int lo = 1, hi = N - 1;
            std::vector<float> yr((size_t)T * N, -2.0f);
            k->fn(w, act.data(), T, yr.data(), lo, hi);
            for (int t = 0; t < T; ++t)
                for (int n = 0; n < N; ++n) {
                    const size_t i = (size_t)t * N + n;
                    CHECK(n >= lo && n < hi ? yr[i] == y[i] : yr[i] == -2.0f);
                }
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

// Every quantizer writes the same bytes as ternary_quant_rows_ref.
static void test_quant_identical() {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    for (int K : {128, 1024, 4096}) {
        const int T = 9;
        std::vector<float> x((size_t)T * K);
        for (auto& v : x) v = u(rng);
        for (int k = 0; k < K; ++k) {
            x[(size_t)1 * K + k] = 0.0f;                                   // all zero
            x[(size_t)2 * K + k] = (float)(k % 255 - 127) + 0.5f * (k & 1); // amax 127, ties at .5
            x[(size_t)3 * K + k] = u(rng) * 1e-39f;                        // denormal
            x[(size_t)4 * K + k] = u(rng) * 3e38f;                         // near FLT_MAX
            x[(size_t)5 * K + k] = k == 7 ? -0.0f : 0.0f;                  // negative zero only
        }
        x[(size_t)2 * K + 5] = 127.0f;
        x[(size_t)6 * K + 3] = NAN;                                        // one NaN
        x[(size_t)7 * K + 3] = INFINITY;                                   // one inf
        x[(size_t)8 * K + K - 1] = -INFINITY;
        x[(size_t)8 * K + 0] = NAN;
        const size_t rb = ternary_act_row_bytes(K);
        std::vector<uint8_t> ref(rb * T, 0xAB);
        ternary_quant_rows_ref(x.data(), K, 0, T, ref.data());
        for (const TernaryQuant* q : ternary_all_quants()) {
            std::vector<uint8_t> a(rb * T, 0xCD);
            q->fn(x.data(), K, 0, 4, a.data());   // split rows, as the op does
            q->fn(x.data(), K, 4, T, a.data());
            for (int t = 0; t < T; ++t)
                if (std::memcmp(&a[(size_t)t * rb], &ref[(size_t)t * rb], rb) != 0) {
                    std::fprintf(stderr, "quant %s differs from the reference (K=%d row %d)\n", q->name, K, t);
                    ++failures;
                }
        }
        std::vector<uint8_t> d(rb * T);
        ternary_quant_rows(x.data(), K, 0, T, d.data());
        CHECK(d == ref);
    }
}

// The repack as it was before the lookup table version: an independent scalar
// reference (per byte v % 3, v /= 3, then bit-plane packing one element at a time).
static void repack_reference(const uint8_t* q, const uint16_t* s, int N, int K, TernaryWeight& out) {
    const int G = K / kTernaryGroup;
    const int nb = (K + 4) / 5;
    out.N = N;
    out.K = K;
    const int B = out.row_blocks();
    out.planes.assign((size_t)B * G * 512, 0);
    out.scales.assign((size_t)B * G * kTernaryRowBlock, 0.0f);
    std::vector<uint8_t> row(K);
    for (int n = 0; n < N; ++n) {
        const int b = n / kTernaryRowBlock, i = n % kTernaryRowBlock;
        for (int g = 0; g < G; ++g)
            out.scales[((size_t)b * G + g) * kTernaryRowBlock + i] = ggml_fp16_to_fp32(s[(size_t)n * G + g]);
        const uint8_t* qr = q + (size_t)n * nb;
        for (int bb = 0; bb < nb; ++bb) {
            int v = qr[bb];
            for (int d = 0; d < 5; ++d) {
                const int k = bb * 5 + d;
                if (k < K) row[k] = (uint8_t)(v % 3);
                v /= 3;
            }
        }
        for (int k = 0; k < K; ++k) {
            const int g = k / kTernaryGroup, r = k % kTernaryGroup;
            const int st = r / 16, p = (r % 16) / 4, j = r % 4;
            out.planes[((size_t)b * G + g) * 512 + (size_t)st * 64 + 4 * i + j] |= (uint8_t)(row[k] << (2 * p));
        }
    }
}

static void test_repack_equivalence() {
    std::mt19937 rng(77);
    int cases = 0;
    for (int N : {1, 2, 3, 5, 15, 16, 17, 31, 32, 33, 37, 64, 80, 1024, 4096})
        for (int K : {128, 256, 384, 1024, 4096}) {
            if ((size_t)N * K > (size_t)1024 * 1024 && K != 1024) continue;  // keep the runtime sane
            const int nb = (K + 4) / 5, G = K / kTernaryGroup;
            // mode 0 random valid, 1 all 0, 2 all 2, 3 all 1, 4 alternating 0/2, 5 any byte 0..242
            for (int mode = 0; mode < 6; ++mode) {
                std::vector<uint8_t> q((size_t)N * nb);
                for (size_t x = 0; x < q.size(); ++x) {
                    int v;
                    switch (mode) {
                        case 0: v = (int)(rng() % 243); break;
                        case 1: v = 0; break;
                        case 2: v = 242; break;  // digits 2 2 2 2 2
                        case 3: v = 121; break;  // digits 1 1 1 1 1
                        case 4: v = (x & 1) ? 242 : 0; break;
                        default: v = (int)(rng() % 243); break;
                    }
                    q[x] = (uint8_t)v;
                }
                std::vector<uint16_t> sc((size_t)N * G);
                for (size_t x = 0; x < sc.size(); ++x) {
                    // random F16 bit patterns, including denormals, large values, inf and nan
                    uint16_t h = (uint16_t)rng();
                    if (x % 7 == 0) h = (uint16_t)(rng() % 0x400);          // denormal
                    else if (x % 11 == 0) h = (uint16_t)(0x7000 + rng() % 0xC00);  // large
                    sc[x] = h;
                }
                TernaryWeight a, b;
                repack_reference(q.data(), sc.data(), N, K, a);
                ternary_repack(q.data(), sc.data(), N, K, b);
                const bool same = a.N == b.N && a.K == b.K && a.planes == b.planes &&
                                  a.scales.size() == b.scales.size() &&
                                  std::memcmp(a.scales.data(), b.scales.data(), a.scales.size() * 4) == 0;
                if (!same) std::fprintf(stderr, "repack mismatch N=%d K=%d mode=%d\n", N, K, mode);
                CHECK(same);
                ++cases;
            }
        }
    std::printf("repack equivalence: %d cases identical to the reference\n", cases);
}

// Bytes 243..255 are not a valid pack of five base-3 digits. The repack must
// refuse them, with the byte value in the message, wherever they sit.
static void test_repack_rejects_invalid_bytes() {
    const int N = 5, K = 256, nb = (K + 4) / 5, G = K / kTernaryGroup;
    const std::vector<uint16_t> sc((size_t)N * G, 0x3C00);  // 1.0
    for (int bad : {243, 250, 255}) {
        for (size_t pos : {(size_t)0, (size_t)(3 * nb + 17), (size_t)(N * nb - 1)}) {
            std::vector<uint8_t> q((size_t)N * nb, 121);
            q[pos] = (uint8_t)bad;
            TernaryWeight w;
            bool threw = false;
            std::string msg;
            try {
                ternary_repack(q.data(), sc.data(), N, K, w);
            } catch (const std::runtime_error& e) {
                threw = true;
                msg = e.what();
            }
            if (!threw || msg.find(std::to_string(bad)) == std::string::npos)
                std::fprintf(stderr, "byte %d at %zu: threw=%d msg=%s\n", bad, pos, (int)threw, msg.c_str());
            CHECK(threw);
            CHECK(msg.find(std::to_string(bad)) != std::string::npos);
        }
    }
    // 242 is the largest valid byte.
    std::vector<uint8_t> q((size_t)N * nb, 242);
    TernaryWeight w;
    ternary_repack(q.data(), sc.data(), N, K, w);
    CHECK(w.N == N && w.K == K);
}

int main() {
    test_repack_equivalence();
    test_repack_rejects_invalid_bytes();
    const Case cases[] = {{8, 128, 1}, {16, 256, 3}, {37, 1024, 5}, {64, 4096, 4}, {5, 384, 7}, {1, 128, 1},
                          {48, 1024, 13}, {80, 512, 200}, {40, 1024, 9, true}, {7, 4096, 3, true}};
    unsigned seed = 1;
    for (const Case& c : cases) run_case(c, seed++);
    // remainder coverage: rows not a multiple of any row tile, columns not a
    // multiple of any column tile
    for (int N : {1, 2, 3, 5, 7, 37})
        for (int T : {1, 2, 3, 5, 7, 9}) run_case({N, 256, T}, seed++);
    test_silence_is_zero();
    test_quant_identical();
    if (failures) return 1;
    std::printf("test_ternary: OK (dispatch=%s; tested:", ternary_kernel_name());
    for (const TernaryKernel* k : ternary_all_kernels()) std::printf(" %s", k->name);
    std::printf("; quant:");
    for (const TernaryQuant* q : ternary_all_quants()) std::printf(" %s", q->name);
    std::printf(")\n");
    return 0;
}
