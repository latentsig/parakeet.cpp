#include "ternary_kernels.hpp"

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#if defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#endif

namespace pk {
namespace {

template <int SH>
inline int8x16_t signed_codes(uint8x16_t v, uint8x16_t m3, int8x16_t one) {
    uint8x16_t c;
    if constexpr (SH == 0) c = vandq_u8(v, m3);
    else c = vandq_u8(vshrq_n_u8(v, SH), m3);
    return vsubq_s8(vreinterpretq_s8_u8(c), one);  // code - 1 in {-1, 0, 1}
}

void rows_neon(const TernaryWeight& w, const uint8_t* act, int T, float* y, int r0, int r1) {
    const int N = w.N, K = w.K, G = w.groups();
    const size_t rb = ternary_act_row_bytes(K);
    const uint8x16_t m3 = vdupq_n_u8(3);
    const int8x16_t one = vdupq_n_s8(1);
    for (int n = r0; n < r1; ++n)
        for (int t = 0; t < T; ++t) {
            const uint8_t* row = act + (size_t)t * rb;
            const int8_t* q = reinterpret_cast<const int8_t*>(row);
            float acc = 0.0f;
            for (int g = 0; g < G; ++g) {
                const uint8_t* p = &w.planes[((size_t)n * G + g) * 32];
                const uint8x16_t v0 = vld1q_u8(p), v1 = vld1q_u8(p + 16);
                const int8_t* x = q + (size_t)g * 128;
                int32x4_t a = vdupq_n_s32(0);
                a = vdotq_s32(a, signed_codes<0>(v0, m3, one), vld1q_s8(x));
                a = vdotq_s32(a, signed_codes<0>(v1, m3, one), vld1q_s8(x + 16));
                a = vdotq_s32(a, signed_codes<2>(v0, m3, one), vld1q_s8(x + 32));
                a = vdotq_s32(a, signed_codes<2>(v1, m3, one), vld1q_s8(x + 48));
                a = vdotq_s32(a, signed_codes<4>(v0, m3, one), vld1q_s8(x + 64));
                a = vdotq_s32(a, signed_codes<4>(v1, m3, one), vld1q_s8(x + 80));
                a = vdotq_s32(a, signed_codes<6>(v0, m3, one), vld1q_s8(x + 96));
                a = vdotq_s32(a, signed_codes<6>(v1, m3, one), vld1q_s8(x + 112));
                acc += w.scales[(size_t)n * G + g] * (float)vaddvq_s32(a);
            }
            y[(size_t)t * N + n] = acc * *reinterpret_cast<const float*>(row + K);
        }
}

}  // namespace

const TernaryKernel* ternary_kernel_neon() {
    static const TernaryKernel k{"neon", rows_neon};
#if defined(__linux__)
    if (!(getauxval(AT_HWCAP) & HWCAP_ASIMDDP)) return nullptr;
#endif
    return &k;
}

}  // namespace pk
#else
namespace pk {
const TernaryKernel* ternary_kernel_neon() { return nullptr; }
}  // namespace pk
#endif
