#include "ternary.hpp"

#include <algorithm>
#include <cmath>

#include "ggml.h"

namespace pk {

void ternary_repack(const uint8_t* q, const uint16_t* s, int N, int K, TernaryWeight& out) {
    GGML_ASSERT(K > 0 && K % kTernaryGroup == 0);
    const int G = K / kTernaryGroup;
    const int nb = (K + 4) / 5;
    out.N = N;
    out.K = K;
    out.planes.assign((size_t)N * G * 32, 0);
    out.scales.resize((size_t)N * G);
    for (size_t i = 0; i < out.scales.size(); ++i) out.scales[i] = ggml_fp16_to_fp32(s[i]);
    std::vector<uint8_t> row(K);
    for (int n = 0; n < N; ++n) {
        const uint8_t* qr = q + (size_t)n * nb;
        for (int b = 0; b < nb; ++b) {
            int v = qr[b];
            for (int d = 0; d < 5; ++d) {
                const int i = b * 5 + d;
                if (i < K) row[i] = (uint8_t)(v % 3);
                v /= 3;
            }
        }
        for (int g = 0; g < G; ++g) {
            uint8_t* p = &out.planes[((size_t)n * G + g) * 32];
            const uint8_t* c = &row[(size_t)g * kTernaryGroup];
            for (int j = 0; j < 32; ++j)
                p[j] = (uint8_t)(c[j] | (c[j + 32] << 2) | (c[j + 64] << 4) | (c[j + 96] << 6));
        }
    }
}

void ternary_dequant(const uint8_t* q, const uint16_t* s, int N, int K, std::vector<float>& W) {
    const int G = (K + kTernaryGroup - 1) / kTernaryGroup;
    const int nb = (K + 4) / 5;
    W.assign((size_t)N * K, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int b = 0; b < nb; ++b) {
            int v = q[(size_t)n * nb + b];
            for (int d = 0; d < 5; ++d) {
                const int i = b * 5 + d;
                if (i < K)
                    W[(size_t)n * K + i] =
                        ggml_fp16_to_fp32(s[(size_t)n * G + i / kTernaryGroup]) * (float)(v % 3 - 1);
                v /= 3;
            }
        }
}

void ternary_quant_rows(const float* x, int K, int t0, int t1, uint8_t* act) {
    const int G = K / kTernaryGroup;
    const size_t rb = ternary_act_row_bytes(K);
    for (int t = t0; t < t1; ++t) {
        const float* xr = x + (size_t)t * K;
        uint8_t* row = act + (size_t)t * rb;
        int8_t* q = reinterpret_cast<int8_t*>(row);
        float* sc = reinterpret_cast<float*>(row + K);
        int32_t* gs = reinterpret_cast<int32_t*>(row + K + 4);
        float amax = 0.0f;
        for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(xr[k]));
        const float s = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        *sc = s;
        for (int g = 0; g < G; ++g) {
            int32_t sum = 0;
            for (int j = 0; j < kTernaryGroup; ++j) {
                int v = (int)std::lrintf(xr[g * kTernaryGroup + j] * inv);
                v = std::min(127, std::max(-127, v));
                q[g * kTernaryGroup + j] = (int8_t)v;
                sum += v;
            }
            gs[g] = sum;
        }
    }
}

void ternary_matmul_rows_ref(const TernaryWeight& w, const uint8_t* act, int T, float* y,
                             int r0, int r1) {
    const int N = w.N, K = w.K, G = w.groups();
    const size_t rb = ternary_act_row_bytes(K);
    for (int n = r0; n < r1; ++n)
        for (int t = 0; t < T; ++t) {
            const uint8_t* row = act + (size_t)t * rb;
            const int8_t* q = reinterpret_cast<const int8_t*>(row);
            const float sa = *reinterpret_cast<const float*>(row + K);
            const int32_t* gs = reinterpret_cast<const int32_t*>(row + K + 4);
            float acc = 0.0f;
            for (int g = 0; g < G; ++g) {
                const uint8_t* p = &w.planes[((size_t)n * G + g) * 32];
                int32_t s = 0;
                for (int j = 0; j < kTernaryGroup; ++j) {
                    const int code = (p[j & 31] >> (2 * (j >> 5))) & 3;
                    s += code * (int32_t)q[g * kTernaryGroup + j];
                }
                acc += w.scales[(size_t)n * G + g] * (float)(s - gs[g]);
            }
            y[(size_t)t * N + n] = acc * sa;
        }
}

// Kernel selection. Later tasks add SIMD kernels here.
void ternary_matmul_rows(const TernaryWeight& w, const uint8_t* act, int T, float* y,
                         int r0, int r1) {
    ternary_matmul_rows_ref(w, act, T, y, r0, r1);
}

const char* ternary_kernel_name() { return "scalar"; }

}  // namespace pk
