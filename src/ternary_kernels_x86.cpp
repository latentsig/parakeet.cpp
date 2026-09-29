#include "ternary_kernels.hpp"

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

#include <algorithm>
#include <cstring>

namespace pk {
namespace {

#define PK_TGT_AVX2 __attribute__((target("avx2")))
#define PK_TGT_512 __attribute__((target("avx2,avx512f,avx512vl,avx512vnni")))

// Both kernels compute the same math as ternary_matmul_rows_ref. One vector
// lane holds one output row of a 16-row block (see TernaryWeight), so the
// per-group integer sums of 16 rows come out in one vector without any
// horizontal reduction. For each output element the float update is exactly
// the reference sequence: acc = acc + scale * (float)(sum - group_sum), in group
// order, then acc * act_scale (separate multiply and add, no FMA).

inline int32_t load_i32(const int8_t* p) {
    int32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

// Row mask of block b for the output range [r0, r1).
inline unsigned block_mask(int b, int r0, int r1) {
    const int lo = std::max(r0 - b * kTernaryRowBlock, 0);
    const int hi = std::min(r1 - b * kTernaryRowBlock, kTernaryRowBlock);
    return hi <= lo ? 0u : ((1u << hi) - 1u) & ~((1u << lo) - 1u);
}

// ---------------------------------------------------------------------------
// AVX-512 VNNI: RB row blocks (16 rows each, one zmm) by C activation rows.
template <int RB, int C>
PK_TGT_512 void tile_512(const TernaryWeight& w, const uint8_t* act, size_t rb, int t0, float* y,
                         int b0, const unsigned* mk) {
    const int N = w.N, K = w.K, G = w.groups();
    const __m512i m3 = _mm512_set1_epi8(3);
    const int8_t* xr[C];
    const int32_t* gsr[C];
    for (int c = 0; c < C; ++c) {
        const uint8_t* row = act + (size_t)(t0 + c) * rb;
        xr[c] = reinterpret_cast<const int8_t*>(row);
        gsr[c] = reinterpret_cast<const int32_t*>(row + K + 4);
    }
    __m512 facc[RB][C];
    for (int b = 0; b < RB; ++b)
        for (int c = 0; c < C; ++c) facc[b][c] = _mm512_setzero_ps();
    for (int g = 0; g < G; ++g) {
        __m512i ia[RB][C];
        for (int b = 0; b < RB; ++b)
            for (int c = 0; c < C; ++c) ia[b][c] = _mm512_setzero_si512();
        const uint8_t* pb[RB];
        for (int b = 0; b < RB; ++b) pb[b] = w.planes.data() + ((size_t)(b0 + b) * G + g) * 512;
        for (int s = 0; s < 8; ++s) {
            __m512i wv[RB][4];
            for (int b = 0; b < RB; ++b) {
                const __m512i v = _mm512_loadu_si512(pb[b] + s * 64);
                wv[b][0] = _mm512_and_si512(v, m3);
                wv[b][1] = _mm512_and_si512(_mm512_srli_epi32(v, 2), m3);
                wv[b][2] = _mm512_and_si512(_mm512_srli_epi32(v, 4), m3);
                wv[b][3] = _mm512_and_si512(_mm512_srli_epi32(v, 6), m3);
            }
            for (int c = 0; c < C; ++c) {
                const int8_t* x = xr[c] + (size_t)g * 128 + s * 16;
                for (int p = 0; p < 4; ++p) {
                    const __m512i a = _mm512_set1_epi32(load_i32(x + 4 * p));
                    for (int b = 0; b < RB; ++b) ia[b][c] = _mm512_dpbusd_epi32(ia[b][c], wv[b][p], a);
                }
            }
        }
        for (int b = 0; b < RB; ++b) {
            const __m512 sc = _mm512_loadu_ps(&w.scales[((size_t)(b0 + b) * G + g) * kTernaryRowBlock]);
            for (int c = 0; c < C; ++c) {
                const __m512 f = _mm512_cvtepi32_ps(_mm512_sub_epi32(ia[b][c], _mm512_set1_epi32(gsr[c][g])));
                facc[b][c] = _mm512_add_ps(facc[b][c], _mm512_mul_ps(sc, f));
            }
        }
    }
    for (int c = 0; c < C; ++c) {
        const __m512 sa = _mm512_set1_ps(*reinterpret_cast<const float*>(xr[c] + K));
        for (int b = 0; b < RB; ++b)
            _mm512_mask_storeu_ps(y + (size_t)(t0 + c) * N + (size_t)(b0 + b) * kTernaryRowBlock,
                                  (__mmask16)mk[b], _mm512_mul_ps(facc[b][c], sa));
    }
}

template <int RB, int C>
PK_TGT_512 void cols_512(const TernaryWeight& w, const uint8_t* act, size_t rb, int T, float* y, int b0,
                         const unsigned* mk) {
    int t0 = 0;
    for (; t0 + C <= T; t0 += C) tile_512<RB, C>(w, act, rb, t0, y, b0, mk);
    for (; t0 < T; ++t0) tile_512<RB, 1>(w, act, rb, t0, y, b0, mk);
}

PK_TGT_512 void rows_512(const TernaryWeight& w, const uint8_t* act, int T, float* y, int r0, int r1) {
    if (r0 >= r1 || T <= 0) return;
    constexpr int RB = 2, C = 4;
    const size_t rb = ternary_act_row_bytes(w.K);
    const int bl = (r1 - 1) / kTernaryRowBlock;
    for (int b = r0 / kTernaryRowBlock; b <= bl;) {
        unsigned mk[RB];
        if (b + RB - 1 <= bl) {
            for (int i = 0; i < RB; ++i) mk[i] = block_mask(b + i, r0, r1);
            cols_512<RB, C>(w, act, rb, T, y, b, mk);
            b += RB;
        } else {
            mk[0] = block_mask(b, r0, r1);
            cols_512<1, C>(w, act, rb, T, y, b, mk);
            b += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2: one row block as two ymm halves (rows 0-7, rows 8-15) by C activation
// rows. maddubs gives int16 pairs of at most 2 * 2 * 127 = 508 in magnitude,
// and a group adds 32 of them per lane, at most 16256, so a whole group
// accumulates exactly in int16 before one widening madd.
template <int C>
PK_TGT_AVX2 void tile_avx2(const TernaryWeight& w, const uint8_t* act, size_t rb, int t0, float* y, int b0,
                           unsigned mk) {
    const int N = w.N, K = w.K, G = w.groups();
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i ones = _mm256_set1_epi16(1);
    const int8_t* xr[C];
    const int32_t* gsr[C];
    for (int c = 0; c < C; ++c) {
        const uint8_t* row = act + (size_t)(t0 + c) * rb;
        xr[c] = reinterpret_cast<const int8_t*>(row);
        gsr[c] = reinterpret_cast<const int32_t*>(row + K + 4);
    }
    alignas(32) float facc[C][16] = {};
    for (int g = 0; g < G; ++g) {
        __m256i ia[C][2];
        for (int c = 0; c < C; ++c) ia[c][0] = ia[c][1] = _mm256_setzero_si256();
        const uint8_t* pb = w.planes.data() + ((size_t)b0 * G + g) * 512;
        for (int s = 0; s < 8; ++s) {
            const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + s * 64));
            const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + s * 64 + 32));
            for (int p = 0; p < 4; ++p) {
                const __m256i w0 = _mm256_and_si256(_mm256_srli_epi32(v0, 2 * p), m3);
                const __m256i w1 = _mm256_and_si256(_mm256_srli_epi32(v1, 2 * p), m3);
                for (int c = 0; c < C; ++c) {
                    const __m256i a = _mm256_set1_epi32(load_i32(xr[c] + (size_t)g * 128 + s * 16 + 4 * p));
                    ia[c][0] = _mm256_add_epi16(ia[c][0], _mm256_maddubs_epi16(w0, a));
                    ia[c][1] = _mm256_add_epi16(ia[c][1], _mm256_maddubs_epi16(w1, a));
                }
            }
        }
        const float* scp = &w.scales[((size_t)b0 * G + g) * kTernaryRowBlock];
        const __m256 sc0 = _mm256_loadu_ps(scp), sc1 = _mm256_loadu_ps(scp + 8);
        for (int c = 0; c < C; ++c) {
            const __m256i gs = _mm256_set1_epi32(gsr[c][g]);
            const __m256 f0 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_madd_epi16(ia[c][0], ones), gs));
            const __m256 f1 = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_madd_epi16(ia[c][1], ones), gs));
            _mm256_store_ps(facc[c], _mm256_add_ps(_mm256_load_ps(facc[c]), _mm256_mul_ps(sc0, f0)));
            _mm256_store_ps(facc[c] + 8, _mm256_add_ps(_mm256_load_ps(facc[c] + 8), _mm256_mul_ps(sc1, f1)));
        }
    }
    for (int c = 0; c < C; ++c) {
        const __m256 sa = _mm256_set1_ps(*reinterpret_cast<const float*>(xr[c] + K));
        alignas(32) float out[16];
        _mm256_store_ps(out, _mm256_mul_ps(_mm256_load_ps(facc[c]), sa));
        _mm256_store_ps(out + 8, _mm256_mul_ps(_mm256_load_ps(facc[c] + 8), sa));
        float* yr = y + (size_t)(t0 + c) * N + (size_t)b0 * kTernaryRowBlock;
        if (mk == 0xFFFFu) std::memcpy(yr, out, sizeof(out));
        else
            for (int i = 0; i < kTernaryRowBlock; ++i)
                if (mk >> i & 1u) yr[i] = out[i];
    }
}

PK_TGT_AVX2 void rows_avx2(const TernaryWeight& w, const uint8_t* act, int T, float* y, int r0, int r1) {
    if (r0 >= r1 || T <= 0) return;
    constexpr int C = 4;
    const size_t rb = ternary_act_row_bytes(w.K);
    const int bl = (r1 - 1) / kTernaryRowBlock;
    for (int b = r0 / kTernaryRowBlock; b <= bl; ++b) {
        const unsigned mk = block_mask(b, r0, r1);
        int t0 = 0;
        for (; t0 + C <= T; t0 += C) tile_avx2<C>(w, act, rb, t0, y, b, mk);
        for (; t0 < T; ++t0) tile_avx2<1>(w, act, rb, t0, y, b, mk);
    }
}

}  // namespace

const TernaryKernel* ternary_kernel_x86_vnni() {
    static const TernaryKernel k{"vnni", rows_512};
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
