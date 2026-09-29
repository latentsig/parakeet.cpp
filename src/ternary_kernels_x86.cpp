#include "ternary_kernels.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

#include <algorithm>

namespace pk {
namespace {

#define PK_TGT_AVX2 __attribute__((target("avx2")))
#define PK_TGT_VNNI __attribute__((target("avx2,avx512f,avx512vl,avx512vnni")))

PK_TGT_AVX2 static inline int32_t hsum_i32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

// u = unsigned weight codes (0..2), s = signed int8 activations, a = int32 acc.
#define PK_DOT_VNNI(a, u, s) _mm256_dpbusd_epi32((a), (u), (s))
#define PK_DOT_AVX2(a, u, s) \
    _mm256_add_epi32((a), _mm256_madd_epi16(_mm256_maddubs_epi16((u), (s)), _mm256_set1_epi16(1)))

// Same math as ternary_matmul_rows_ref: per group, exact int32 sum of
// code * activation, then acc += scale * (float)(sum - group_sum). Weights are
// unpacked once per (row, group) and reused for up to 4 activation rows.
#define PK_TERNARY_KERNEL(NAME, TGT, DOT)                                                     \
    TGT static void NAME(const TernaryWeight& w, const uint8_t* act, int T, float* y,         \
                         int r0, int r1) {                                                    \
        const int N = w.N, K = w.K, G = w.groups();                                           \
        const size_t rb = ternary_act_row_bytes(K);                                           \
        const __m256i m3 = _mm256_set1_epi8(3);                                               \
        for (int n = r0; n < r1; ++n) {                                                       \
            for (int t0 = 0; t0 < T; t0 += 4) {                                               \
                const int nt = std::min(4, T - t0);                                           \
                float acc[4] = {0.f, 0.f, 0.f, 0.f};                                          \
                for (int g = 0; g < G; ++g) {                                                 \
                    const __m256i v = _mm256_loadu_si256(                                     \
                        reinterpret_cast<const __m256i*>(&w.planes[((size_t)n * G + g) * 32]));\
                    const __m256i p0 = _mm256_and_si256(v, m3);                               \
                    const __m256i p1 = _mm256_and_si256(_mm256_srli_epi16(v, 2), m3);         \
                    const __m256i p2 = _mm256_and_si256(_mm256_srli_epi16(v, 4), m3);         \
                    const __m256i p3 = _mm256_and_si256(_mm256_srli_epi16(v, 6), m3);         \
                    const float sc = w.scales[(size_t)n * G + g];                             \
                    for (int c = 0; c < nt; ++c) {                                            \
                        const uint8_t* row = act + (size_t)(t0 + c) * rb;                     \
                        const int8_t* x = reinterpret_cast<const int8_t*>(row) + (size_t)g * 128; \
                        const int32_t gs = reinterpret_cast<const int32_t*>(row + K + 4)[g];  \
                        __m256i a = _mm256_setzero_si256();                                   \
                        a = DOT(a, p0, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x)));      \
                        a = DOT(a, p1, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + 32))); \
                        a = DOT(a, p2, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + 64))); \
                        a = DOT(a, p3, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + 96))); \
                        acc[c] += sc * (float)(hsum_i32(a) - gs);                             \
                    }                                                                         \
                }                                                                             \
                for (int c = 0; c < nt; ++c) {                                                \
                    const uint8_t* row = act + (size_t)(t0 + c) * rb;                         \
                    y[(size_t)(t0 + c) * N + n] = acc[c] * *reinterpret_cast<const float*>(row + K); \
                }                                                                             \
            }                                                                                 \
        }                                                                                     \
    }

PK_TERNARY_KERNEL(rows_vnni, PK_TGT_VNNI, PK_DOT_VNNI)
PK_TERNARY_KERNEL(rows_avx2, PK_TGT_AVX2, PK_DOT_AVX2)

}  // namespace

const TernaryKernel* ternary_kernel_x86_vnni() {
    static const TernaryKernel k{"vnni", rows_vnni};
    __builtin_cpu_init();
    const bool ok = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl") &&
                    __builtin_cpu_supports("avx512vnni");
    return ok ? &k : nullptr;
}

const TernaryKernel* ternary_kernel_x86_avx2() {
    static const TernaryKernel k{"avx2", rows_avx2};
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") ? &k : nullptr;
}

}  // namespace pk
#else
namespace pk {
const TernaryKernel* ternary_kernel_x86_vnni() { return nullptr; }
const TernaryKernel* ternary_kernel_x86_avx2() { return nullptr; }
}  // namespace pk
#endif
