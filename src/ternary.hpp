#pragma once
// Native ternary (1.58 bit) linear layers for moondream/parakeet-redux.
// See docs/superpowers/specs/2026-09-29-hf-ternary-vad-design.md.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace pk {

class ModelLoader;

constexpr int kTernaryGroup = 128;

// Repacked weight, N output rows by K input columns (K % 128 == 0).
// planes: per row, K/128 groups of 32 bytes. Byte j of a group holds the codes
// (0,1,2) of elements j, j+32, j+64, j+96 in bit pairs 0-1, 2-3, 4-5, 6-7.
// w[n][k] = scales[n][k/128] * (code - 1).
struct TernaryWeight {
    int N = 0;
    int K = 0;
    std::vector<uint8_t> planes;
    std::vector<float>   scales;   // N * groups(), row-major
    int groups() const { return K / kTernaryGroup; }
};

// Upstream packing (5 trits per byte, base 3, LSD first) plus F16 group scales
// to the kernel layout. qweight is N rows of ceil(K/5) bytes.
void ternary_repack(const uint8_t* qweight, const uint16_t* scales_f16, int N, int K,
                    TernaryWeight& out);
// Float reference dequantization, W becomes N*K row-major.
void ternary_dequant(const uint8_t* qweight, const uint16_t* scales_f16, int N, int K,
                     std::vector<float>& W);

// Quantized activations, one row per token: K int8 values, one float scale,
// K/128 int32 group sums of the int8 values.
inline size_t ternary_act_row_bytes(int K) {
    return (size_t)K + 4 + 4 * (size_t)(K / kTernaryGroup);
}
// x is the full [T][K] float input; writes rows [t0, t1) of act.
void ternary_quant_rows(const float* x, int K, int t0, int t1, uint8_t* act);

// y[t*N + n] for n in [r0, r1) and all t < T. The reference defines correctness;
// the dispatching entry point picks the best kernel for this CPU.
void ternary_matmul_rows_ref(const TernaryWeight& w, const uint8_t* act, int T, float* y,
                             int r0, int r1);
void ternary_matmul_rows(const TernaryWeight& w, const uint8_t* act, int T, float* y,
                         int r0, int r1);
const char* ternary_kernel_name();

// True iff the GGUF holds <base>.qweight (a packed ternary linear).
bool has_ternary(const ModelLoader& ml, const std::string& base);
// nn.Linear over x (f32, ne = [K, ...]) with the packed weight <base>.qweight /
// <base>.scales. Returns f32 with ne = [N, ...]. Bias is the caller's job.
ggml_tensor* ternary_linear(ggml_context* ctx, const ModelLoader& ml, const std::string& base,
                            ggml_tensor* x);

}  // namespace pk
