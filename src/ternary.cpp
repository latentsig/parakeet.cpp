#include "ternary.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

#include "ggml.h"
#include "common.hpp"
#include "model_loader.hpp"
#include "ternary_kernels.hpp"

namespace pk {

void ternary_repack(const uint8_t* q, const uint16_t* s, int N, int K, TernaryWeight& out) {
    GGML_ASSERT(K > 0 && K % kTernaryGroup == 0);
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
                int32_t s = 0;
                for (int j = 0; j < kTernaryGroup; ++j)
                    s += w.code(n, g * kTernaryGroup + j) * (int32_t)q[g * kTernaryGroup + j];
                acc += w.scale(n, g) * (float)(s - gs[g]);
            }
            y[(size_t)t * N + n] = acc * sa;
        }
}

namespace {

const TernaryKernel kScalarKernel{"scalar", ternary_matmul_rows_ref};

const TernaryKernel* pick_kernel() {
    static const TernaryKernel* chosen = [] {
        if (const char* e = std::getenv("PARAKEET_TERNARY_KERNEL")) {
            const TernaryKernel* want = nullptr;
            if (!std::strcmp(e, "scalar")) want = &kScalarKernel;
            else if (!std::strcmp(e, "avx2")) want = ternary_kernel_x86_avx2();
            else if (!std::strcmp(e, "vnni")) want = ternary_kernel_x86_vnni();
            else if (!std::strcmp(e, "neon")) want = ternary_kernel_neon();
            if (want) return want;
            PK_LOG("PARAKEET_TERNARY_KERNEL=%s is unavailable here; using automatic selection", e);
        }
        if (const TernaryKernel* k = ternary_kernel_x86_vnni()) return k;
        if (const TernaryKernel* k = ternary_kernel_neon()) return k;
        if (const TernaryKernel* k = ternary_kernel_x86_avx2()) return k;
        return &kScalarKernel;
    }();
    return chosen;
}

}  // namespace

std::vector<const TernaryKernel*> ternary_all_kernels() {
    std::vector<const TernaryKernel*> v{&kScalarKernel};
    for (const TernaryKernel* k : {ternary_kernel_x86_avx2(), ternary_kernel_x86_vnni(), ternary_kernel_neon()})
        if (k) v.push_back(k);
    return v;
}

void ternary_matmul_rows(const TernaryWeight& w, const uint8_t* act, int T, float* y, int r0, int r1) {
    pick_kernel()->fn(w, act, T, y, r0, r1);
}

const char* ternary_kernel_name() { return pick_kernel()->name; }

namespace {

struct TernaryStore {
    std::mutex mu;
    std::unordered_map<std::string, std::unique_ptr<TernaryWeight>> weights;
};

const TernaryWeight& weight_for(const ModelLoader& ml, const std::string& base) {
    static std::mutex init_mu;
    TernaryStore* store;
    {
        std::lock_guard<std::mutex> lk(init_mu);
        auto& slot = ml.ternary_store();
        if (!slot) slot = std::make_shared<TernaryStore>();
        store = static_cast<TernaryStore*>(slot.get());
    }
    std::lock_guard<std::mutex> lk(store->mu);
    auto it = store->weights.find(base);
    if (it != store->weights.end()) return *it->second;
    const ggml_tensor* q = ml.tensor(base + ".qweight");
    const ggml_tensor* s = ml.tensor(base + ".scales");
    if (!q || !s) throw std::runtime_error("ternary: missing " + base + ".qweight/.scales");
    const int N = (int)q->ne[1];
    const int K = (int)s->ne[0] * kTernaryGroup;
    if ((int)q->ne[0] != (K + 4) / 5 || (int)s->ne[1] != N)
        throw std::runtime_error("ternary: inconsistent shapes for " + base);
    auto w = std::make_unique<TernaryWeight>();
    ternary_repack(static_cast<const uint8_t*>(q->data), static_cast<const uint16_t*>(s->data), N, K, *w);
    return *store->weights.emplace(base, std::move(w)).first->second;
}

// Op 1: per-row int8 quantization of the activations. dst is I8, ne0 = row bytes.
void op_quant(ggml_tensor* dst, int ith, int nth, void*) {
    const ggml_tensor* x = dst->src[0];
    const int K = (int)x->ne[0];
    const int T = (int)ggml_nrows(x);
    const int t0 = (int)((int64_t)T * ith / nth);
    const int t1 = (int)((int64_t)T * (ith + 1) / nth);
    ternary_quant_rows(static_cast<const float*>(x->data), K, t0, t1, static_cast<uint8_t*>(dst->data));
}

// Op 2: ternary matmul. src0 is the quantized activation tensor from op 1,
// userdata is the TernaryWeight. Threads split the output rows.
void op_matmul(ggml_tensor* dst, int ith, int nth, void* ud) {
    const TernaryWeight& w = *static_cast<const TernaryWeight*>(ud);
    const int T = (int)ggml_nrows(dst);
    const int r0 = (int)((int64_t)w.N * ith / nth);
    const int r1 = (int)((int64_t)w.N * (ith + 1) / nth);
    ternary_matmul_rows(w, static_cast<const uint8_t*>(dst->src[0]->data), T,
                        static_cast<float*>(dst->data), r0, r1);
}

}  // namespace

bool has_ternary(const ModelLoader& ml, const std::string& base) {
    return ml.tensor(base + ".qweight") != nullptr;
}

ggml_tensor* ternary_linear(ggml_context* ctx, const ModelLoader& ml, const std::string& base,
                            ggml_tensor* x) {
    const TernaryWeight& w = weight_for(ml, base);
    if (x->type != GGML_TYPE_F32 || (int)x->ne[0] != w.K)
        throw std::runtime_error("ternary: input shape mismatch for " + base);
    if (!ggml_is_contiguous(x)) x = ggml_cont(ctx, x);
    const int64_t T = ggml_nrows(x);
    ggml_tensor* a1[1] = {x};
    ggml_tensor* act = ggml_custom_4d(ctx, GGML_TYPE_I8, (int64_t)ternary_act_row_bytes(w.K), T, 1, 1,
                                      a1, 1, op_quant, GGML_N_TASKS_MAX, nullptr);
    ggml_tensor* a2[1] = {act};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, w.N, x->ne[1], x->ne[2], x->ne[3], a2, 1, op_matmul,
                          GGML_N_TASKS_MAX, const_cast<TernaryWeight*>(&w));
}

void ternary_prepare(const ModelLoader& ml) {
    static const char* const kLinears[] = {
        "feed_forward1.linear1", "feed_forward1.linear2", "feed_forward2.linear1",
        "feed_forward2.linear2", "self_attn.linear_q",    "self_attn.linear_k",
        "self_attn.linear_v",    "self_attn.linear_out",  "self_attn.linear_pos",
        "conv.pointwise_conv1",  "conv.pointwise_conv2"};
    const int n_layers = (int)ml.config().n_layers;
    for (int i = 0; i < n_layers; ++i) {
        for (const char* nm : kLinears) {
            const std::string base = "encoder.layers." + std::to_string(i) + "." + nm;
            if (has_ternary(ml, base)) {
                weight_for(ml, base);
            } else if (!ml.tensor(base + ".weight")) {
                throw std::runtime_error("ternary: missing weight " + base);
            }
        }
    }
}

}  // namespace pk
