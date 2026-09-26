#include "diarization_encoder.hpp"
#include "backend.hpp"
#include "graph_builder.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <string>
#include <vector>

namespace pk {

// ============================================================================
// DiarizationEncoder — pre-LN RoPE Transformer for Nemotron-3-Diarization.
//
//   mel [n_mels=128, T]
//     → FeatureStacking: transpose, pad, reshape [1024, T/8], Linear(1024→512)
//     → embed_norm: LayerNorm(512, eps=1e-5)
//     → 31× TransformerBlock (pre-norm):
//         x = x + attn(norm1(x))
//         x = x + ffn(norm2(x))
//     → final_norm: LayerNorm(512, eps=1e-5)
//     → transpose → output [d_model, T_enc] (channels-first: enc_out[c*Tp+t])
// ============================================================================

DiarizationEncoder::DiarizationEncoder(const ModelLoader& ml)
    : ml_(ml) {
    const auto& cfg = ml.config();
    d_model_            = (int)cfg.d_model;
    n_layers_           = (int)cfg.n_layers;
    n_heads_            = (int)cfg.n_heads;
    head_dim_           = d_model_ / n_heads_;
    ff_dim_             = (int)cfg.ff_dim;
    subsampling_factor_ = (int)cfg.subsampling_factor;
    n_mels_             = (int)cfg.n_mels;
    qkv_bias_           = cfg.use_bias;
    pre_block_norm_     = true;
    rope_base_          = 10000.0f;
    rotary_fraction_    = 1.0f;
    ln_eps_             = 1e-5f;

    assert(n_layers_ > 0 && d_model_ > 0);
    assert(subsampling_factor_ > 0);
    assert(head_dim_ * n_heads_ == d_model_);
}

void DiarizationEncoder::forward(const std::vector<float>& mel, int n_mels, int T,
                                  std::vector<float>& enc_out,
                                  int& d_model, int& T_enc) const {
    assert(n_mels == n_mels_);
    assert((int)mel.size() == n_mels * T);

    const int factor = subsampling_factor_;  // 8
    const int pad = (factor - (T % factor)) % factor;
    const int T_padded = T + pad;
    const int Tp = T_padded / factor;

    std::vector<int32_t> positions(Tp);
    for (int i = 0; i < Tp; ++i) positions[i] = i;

    const ModelLoader& ml = ml_;
    const int d = d_model_;
    const int H = n_heads_;
    const int hd = head_dim_;
    const int nls = n_layers_;
    const float ln_eps = ln_eps_;
    const float rope_base = rope_base_;
    const int n_rot = (int)(hd * rotary_fraction_);

    pk::ensure_weights_realized(ml);
    GraphInputPool pool;

    bool ok = pk::run_graph(0, 0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // --- 1. FeatureStacking ---
            // mel is [n_mels, T] row-major: mel[m*T + t]
            // Copy into padded buffer [n_mels, T_padded]
            int64_t mel_ne[2] = {T_padded, n_mels};
            std::vector<float>& mel_padded = pool.alloc_f32((size_t)n_mels * T_padded);
            for (int m = 0; m < n_mels; ++m)
                for (int t = 0; t < T; ++t)
                    mel_padded[(size_t)m * T_padded + t] = mel[(size_t)m * T + t];

            ggml_tensor* mel_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mel_ne,
                                  mel_padded.data(), (size_t)n_mels * T_padded * sizeof(float));
            mel_t = ggml_cont(ctx, mel_t);
            // mel_t: ne[0]=T_padded, ne[1]=n_mels

            // NeMo FeatureStacking transposes [C, T] → [T, C] before reshape.
            // After transpose: ne[0]=n_mels, ne[1]=T_padded (time-major)
            mel_t = ggml_cont(ctx, ggml_transpose(ctx, mel_t));

            // Reshape [T_padded, n_mels] → [n_mels*factor, Tp] = [1024, Tp]
            ggml_tensor* stacked = ggml_reshape_2d(ctx, mel_t, (int64_t)n_mels * factor, Tp);

            // Linear(1024 → 512, no bias)
            ggml_tensor* proj_w = pk::clone_weight(ctx, ml, "encoder.pre_encode.proj.weight");
            ggml_tensor* x = ggml_mul_mat(ctx, proj_w, stacked);  // [d_model, Tp]

            // --- 2. embed_norm ---
            {
                ggml_tensor* g = pk::clone_weight(ctx, ml, "encoder.embed_norm.weight");
                ggml_tensor* b = pk::clone_weight(ctx, ml, "encoder.embed_norm.bias");
                ggml_tensor* y = ggml_norm(ctx, x, ln_eps);
                x = ggml_add(ctx, ggml_mul(ctx, y, g), b);
            }

            // --- 3. Position tensor for RoPE ---
            int64_t pos_ne[1] = {Tp};
            ggml_tensor* pos = pk::graph_input_tensor(ctx, GGML_TYPE_I32, 1, pos_ne,
                                  positions.data(), (size_t)Tp * sizeof(int32_t));

            // --- 4. N × TransformerBlock (pre-norm) ---
            for (int i = 0; i < nls; ++i) {
                std::string base = "encoder.layers." + std::to_string(i) + ".";

                // norm1
                {
                    ggml_tensor* g = pk::clone_weight(ctx, ml, (base + "norm1.weight").c_str());
                    ggml_tensor* b = pk::clone_weight(ctx, ml, (base + "norm1.bias").c_str());
                    ggml_tensor* h = ggml_norm(ctx, x, ln_eps);
                    h = ggml_add(ctx, ggml_mul(ctx, h, g), b);

                    // Fused QKV (no bias)
                    ggml_tensor* qkv_w = pk::clone_weight(ctx, ml, (base + "attn.w_qkv.weight").c_str());
                    ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, h);  // [3*d, Tp]
                    qkv = ggml_cont(ctx, qkv);
                    qkv = ggml_reshape_3d(ctx, qkv, d, 3, Tp);  // [d, 3, Tp]

                    // Split Q, K, V
                    size_t ts = (size_t)3 * d * sizeof(float);
                    ggml_tensor* q = ggml_view_2d(ctx, qkv, d, Tp, ts, 0);
                    ggml_tensor* k = ggml_view_2d(ctx, qkv, d, Tp, ts, (size_t)d * sizeof(float));
                    ggml_tensor* v = ggml_view_2d(ctx, qkv, d, Tp, ts, (size_t)2 * d * sizeof(float));

                    // Reshape to [hd, H, Tp]
                    q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), hd, H, Tp);
                    k = ggml_reshape_3d(ctx, ggml_cont(ctx, k), hd, H, Tp);
                    v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), hd, H, Tp);

                    // RoPE (GPT-NeoX)
                    q = ggml_rope_ext(ctx, q, pos, nullptr, n_rot,
                            GGML_ROPE_TYPE_NEOX, 0, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                    k = ggml_rope_ext(ctx, k, pos, nullptr, n_rot,
                            GGML_ROPE_TYPE_NEOX, 0, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

                    // Permute to [hd, Tp, H] (head in batch dim)
                    q = ggml_permute(ctx, q, 0, 2, 1, 3);
                    k = ggml_permute(ctx, k, 0, 2, 1, 3);
                    v = ggml_permute(ctx, v, 0, 2, 1, 3);

                    // flash_attn_ext: q [hd, Tp, H], k [hd, Tp, H], v [hd, Tp, H]
                    // Result: ne = [hd, H, Tp, 1] (ggml.c line 5357-5358)
                    // Memory order: flat[t*H*hd + h*hd + d] = [T, H, hd]
                    // This is ALREADY the correct PyTorch merge order
                    // (attn.transpose(1,2).contiguous().view(B,T,d_model)).
                    // Do NOT permute — just cont + reshape.
                    float scale = 1.0f / std::sqrt((float)hd);
                    ggml_tensor* attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
                                                             scale, 0.0f, 0.0f);
                    attn = ggml_cont(ctx, attn);
                    attn = ggml_reshape_2d(ctx, attn, (int64_t)d, (int64_t)Tp);

                    // out_proj (with bias)
                    ggml_tensor* op_w = pk::clone_weight(ctx, ml, (base + "attn.out_proj.weight").c_str());
                    attn = ggml_mul_mat(ctx, op_w, attn);
                    ggml_tensor* op_b = pk::clone_weight_opt(ctx, ml, (base + "attn.out_proj.bias").c_str());
                    if (op_b) attn = ggml_add(ctx, attn, op_b);

                    // Residual
                    x = ggml_add(ctx, x, attn);
                }

                // norm2 + FFN
                {
                    ggml_tensor* g = pk::clone_weight(ctx, ml, (base + "norm2.weight").c_str());
                    ggml_tensor* b = pk::clone_weight(ctx, ml, (base + "norm2.bias").c_str());
                    ggml_tensor* h = ggml_norm(ctx, x, ln_eps);
                    h = ggml_add(ctx, ggml_mul(ctx, h, g), b);

                    // FFN: Linear(d→ff, bias) → GELU → Linear(ff→d, bias)
                    ggml_tensor* f0_w = pk::clone_weight(ctx, ml, (base + "ffn.net.0.weight").c_str());
                    h = ggml_mul_mat(ctx, f0_w, h);
                    ggml_tensor* f0_b = pk::clone_weight_opt(ctx, ml, (base + "ffn.net.0.bias").c_str());
                    if (f0_b) h = ggml_add(ctx, h, f0_b);
                    h = ggml_gelu(ctx, h);

                    ggml_tensor* f3_w = pk::clone_weight(ctx, ml, (base + "ffn.net.3.weight").c_str());
                    h = ggml_mul_mat(ctx, f3_w, h);
                    ggml_tensor* f3_b = pk::clone_weight_opt(ctx, ml, (base + "ffn.net.3.bias").c_str());
                    if (f3_b) h = ggml_add(ctx, h, f3_b);

                    x = ggml_add(ctx, x, h);
                }
            }

            // --- 5. final_norm ---
            {
                ggml_tensor* g = pk::clone_weight(ctx, ml, "encoder.final_norm.weight");
                ggml_tensor* b = pk::clone_weight(ctx, ml, "encoder.final_norm.bias");
                ggml_tensor* y = ggml_norm(ctx, x, ln_eps);
                x = ggml_add(ctx, ggml_mul(ctx, y, g), b);
            }

            // --- 6. Transpose to channels-first ---
            // x: [d_model, Tp] (ne[0]=d, ne[1]=Tp)
            // ggml flat: [d0_t0, d1_t0, ..., d511_t0, d0_t1, ...] = time-major
            // We need channels-first: enc_out[c*Tp + t]
            // transpose → [Tp, d_model], cont → flat[c*Tp + t] ✓
            x = ggml_cont(ctx, ggml_transpose(ctx, x));
            return x;
        }, enc_out);

    assert(ok && "diarization encoder graph failed");
    (void)ok;

    d_model = d_model_;
    T_enc = Tp;
}

// ============================================================================
// Streaming split: pre_encode + transformer_forward
// ============================================================================

void DiarizationEncoder::pre_encode(const std::vector<float>& mel, int n_mels, int T,
                                    std::vector<float>& emb, int& d_model, int& T_enc) const {
    assert(n_mels == n_mels_);
    assert((int)mel.size() == n_mels * T);

    const int factor = subsampling_factor_;
    const int pad = (factor - (T % factor)) % factor;
    const int T_padded = T + pad;
    const int Tp = T_padded / factor;

    const ModelLoader& ml = ml_;
    const int d = d_model_;
    const float ln_eps = ln_eps_;

    pk::ensure_weights_realized(ml);
    GraphInputPool pool;

    bool ok = pk::run_graph(0, 0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // --- 1. FeatureStacking ---
            int64_t mel_ne[2] = {T_padded, n_mels};
            std::vector<float>& mel_padded = pool.alloc_f32((size_t)n_mels * T_padded);
            for (int m = 0; m < n_mels; ++m)
                for (int t = 0; t < T; ++t)
                    mel_padded[(size_t)m * T_padded + t] = mel[(size_t)m * T + t];

            ggml_tensor* mel_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mel_ne,
                                  mel_padded.data(), (size_t)n_mels * T_padded * sizeof(float));
            mel_t = ggml_cont(ctx, mel_t);
            mel_t = ggml_cont(ctx, ggml_transpose(ctx, mel_t));
            ggml_tensor* stacked = ggml_reshape_2d(ctx, mel_t, (int64_t)n_mels * factor, Tp);

            // Linear(1024 → 512, no bias)
            ggml_tensor* proj_w = pk::clone_weight(ctx, ml, "encoder.pre_encode.proj.weight");
            ggml_tensor* x = ggml_mul_mat(ctx, proj_w, stacked);

            // --- 2. embed_norm ---
            ggml_tensor* g = pk::clone_weight(ctx, ml, "encoder.embed_norm.weight");
            ggml_tensor* b = pk::clone_weight(ctx, ml, "encoder.embed_norm.bias");
            ggml_tensor* y = ggml_norm(ctx, x, ln_eps);
            x = ggml_add(ctx, ggml_mul(ctx, y, g), b);

            // Transpose to channels-first: [d_model, T_enc]
            // pre_encode returns channels-first: emb[c*T_enc + t]
            x = ggml_cont(ctx, ggml_transpose(ctx, x));
            return x;
        }, emb);

    assert(ok && "diarization pre_encode graph failed");
    (void)ok;

    d_model = d_model_;
    T_enc = Tp;
}

void DiarizationEncoder::transformer_forward(const std::vector<float>& emb, int d_model, int T_enc,
                                              std::vector<float>& enc_out) const {
    assert(d_model == d_model_);
    assert((int)emb.size() == d_model * T_enc);

    const ModelLoader& ml = ml_;
    const int d = d_model_;
    const int H = n_heads_;
    const int hd = head_dim_;
    const int nls = n_layers_;
    const float ln_eps = ln_eps_;
    const float rope_base = rope_base_;
    const int n_rot = (int)(hd * rotary_fraction_);

    std::vector<int32_t> positions(T_enc);
    for (int i = 0; i < T_enc; ++i) positions[i] = i;

    pk::ensure_weights_realized(ml);
    GraphInputPool pool;

    bool ok = pk::run_graph(0, 0,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // Input: emb is channels-first [d_model, T_enc] → emb[c*T_enc + t]
            // ggml column-major: ne[0]=T_enc (fastest), ne[1]=d_model
            // → flat[t + c*T_enc] = emb[c*T_enc + t] ✓
            int64_t emb_ne[2] = {T_enc, d};
            ggml_tensor* x = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, emb_ne,
                                  const_cast<float*>(emb.data()), (size_t)d * T_enc * sizeof(float));
            // Transpose to time-major: ne[0]=d_model, ne[1]=T_enc
            // (same layout as forward() uses after pre-encode)
            x = ggml_cont(ctx, ggml_transpose(ctx, x));
            // x: ne[0]=d, ne[1]=T_enc — time-major

            // Position tensor for RoPE
            int64_t pos_ne[1] = {(int64_t)T_enc};
            ggml_tensor* pos = pk::graph_input_tensor(ctx, GGML_TYPE_I32, 1, pos_ne,
                                  positions.data(), (size_t)T_enc * sizeof(int32_t));

            // N × TransformerBlock (pre-norm)
            for (int i = 0; i < nls; ++i) {
                std::string base = "encoder.layers." + std::to_string(i) + ".";

                // norm1
                {
                    ggml_tensor* g = pk::clone_weight(ctx, ml, (base + "norm1.weight").c_str());
                    ggml_tensor* b = pk::clone_weight(ctx, ml, (base + "norm1.bias").c_str());
                    ggml_tensor* h = ggml_norm(ctx, x, ln_eps);
                    h = ggml_add(ctx, ggml_mul(ctx, h, g), b);

                    // Fused QKV (no bias)
                    ggml_tensor* qkv_w = pk::clone_weight(ctx, ml, (base + "attn.w_qkv.weight").c_str());
                    ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, h);
                    qkv = ggml_cont(ctx, qkv);
                    qkv = ggml_reshape_3d(ctx, qkv, d, 3, T_enc);

                    size_t ts = (size_t)3 * d * sizeof(float);
                    ggml_tensor* q = ggml_view_2d(ctx, qkv, d, T_enc, ts, 0);
                    ggml_tensor* k = ggml_view_2d(ctx, qkv, d, T_enc, ts, (size_t)d * sizeof(float));
                    ggml_tensor* v = ggml_view_2d(ctx, qkv, d, T_enc, ts, (size_t)2 * d * sizeof(float));

                    q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), hd, H, T_enc);
                    k = ggml_reshape_3d(ctx, ggml_cont(ctx, k), hd, H, T_enc);
                    v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), hd, H, T_enc);

                    q = ggml_rope_ext(ctx, q, pos, nullptr, n_rot,
                            GGML_ROPE_TYPE_NEOX, 0, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                    k = ggml_rope_ext(ctx, k, pos, nullptr, n_rot,
                            GGML_ROPE_TYPE_NEOX, 0, rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

                    q = ggml_permute(ctx, q, 0, 2, 1, 3);
                    k = ggml_permute(ctx, k, 0, 2, 1, 3);
                    v = ggml_permute(ctx, v, 0, 2, 1, 3);

                    float scale = 1.0f / std::sqrt((float)hd);
                    ggml_tensor* attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr,
                                                             scale, 0.0f, 0.0f);
                    attn = ggml_cont(ctx, attn);
                    attn = ggml_reshape_2d(ctx, attn, (int64_t)d, (int64_t)T_enc);

                    ggml_tensor* op_w = pk::clone_weight(ctx, ml, (base + "attn.out_proj.weight").c_str());
                    attn = ggml_mul_mat(ctx, op_w, attn);
                    ggml_tensor* op_b = pk::clone_weight_opt(ctx, ml, (base + "attn.out_proj.bias").c_str());
                    if (op_b) attn = ggml_add(ctx, attn, op_b);

                    x = ggml_add(ctx, x, attn);
                }

                // norm2 + FFN
                {
                    ggml_tensor* g = pk::clone_weight(ctx, ml, (base + "norm2.weight").c_str());
                    ggml_tensor* b = pk::clone_weight(ctx, ml, (base + "norm2.bias").c_str());
                    ggml_tensor* h = ggml_norm(ctx, x, ln_eps);
                    h = ggml_add(ctx, ggml_mul(ctx, h, g), b);

                    ggml_tensor* f0_w = pk::clone_weight(ctx, ml, (base + "ffn.net.0.weight").c_str());
                    h = ggml_mul_mat(ctx, f0_w, h);
                    ggml_tensor* f0_b = pk::clone_weight_opt(ctx, ml, (base + "ffn.net.0.bias").c_str());
                    if (f0_b) h = ggml_add(ctx, h, f0_b);
                    h = ggml_gelu(ctx, h);

                    ggml_tensor* f3_w = pk::clone_weight(ctx, ml, (base + "ffn.net.3.weight").c_str());
                    h = ggml_mul_mat(ctx, f3_w, h);
                    ggml_tensor* f3_b = pk::clone_weight_opt(ctx, ml, (base + "ffn.net.3.bias").c_str());
                    if (f3_b) h = ggml_add(ctx, h, f3_b);

                    x = ggml_add(ctx, x, h);
                }
            }

            // final_norm
            {
                ggml_tensor* g = pk::clone_weight(ctx, ml, "encoder.final_norm.weight");
                ggml_tensor* b = pk::clone_weight(ctx, ml, "encoder.final_norm.bias");
                ggml_tensor* y = ggml_norm(ctx, x, ln_eps);
                x = ggml_add(ctx, ggml_mul(ctx, y, g), b);
            }

            // Transpose to channels-first
            x = ggml_cont(ctx, ggml_transpose(ctx, x));
            return x;
        }, enc_out);

    assert(ok && "diarization transformer_forward graph failed");
    (void)ok;
}

} // namespace pk
