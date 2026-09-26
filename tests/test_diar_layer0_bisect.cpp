// test_diar_layer0_bisect.cpp — dump layer-0 intermediates stage by stage
// for comparison with PyTorch reference (dump_layer0_ref.py).
#include "diarization.hpp"
#include "mel.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "audio_io.hpp"
#include "diarization_encoder.hpp"
#include "diarization_head.hpp"
#include "ggml_graph.hpp"
#include "graph_builder.hpp"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <vector>

static void write_npy_f32(const std::string& path, const float* data,
                          const std::vector<int64_t>& shape) {
    std::string magic = "\x93NUMPY";
    uint8_t version[2] = {1, 0};
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) dict += ", ";
        dict += std::to_string(shape[i]);
    }
    if (shape.size() == 1) dict += ",";
    dict += "), }";
    int overhead = 10;
    int target = overhead + dict.size() + 1;
    int padded = ((target + 63) / 64) * 64;
    int n_pad = padded - target;
    uint16_t hlen = (uint16_t)(dict.size() + n_pad + 1);
    std::ofstream f(path, std::ios::binary);
    f.write(magic.data(), 6);
    f.write((char*)version, 2);
    f.write((char*)&hlen, 2);
    f.write(dict.data(), (std::streamsize)dict.size());
    for (int i = 0; i < n_pad; ++i) f.write(" ", 1);
    f.write("\n", 1);
    size_t n = 1;
    for (auto s : shape) n *= s;
    f.write((const char*)data, (std::streamsize)(n * sizeof(float)));
}

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    const char* wav_path = std::getenv("PARAKEET_TEST_DIAR_WAV");
    const char* out_dir = std::getenv("PARAKEET_TEST_DIAR_OUT");
    if (!gguf || !wav_path || !out_dir) return 77;

    auto m = pk::DiarizationModel::load(gguf);
    if (!m) return 1;

    pk::Audio audio;
    if (!pk::load_audio_16k_mono(wav_path, audio)) return 1;

    const pk::ModelLoader& ml = m->loader();
    pk::MelFrontend mel(ml);

    std::vector<float> feats;
    int n_mels = 0, T = 0;
    mel.compute(audio.samples, feats, n_mels, T);

    const auto& cfg = ml.config();
    int d_model = (int)cfg.d_model;
    int n_heads = (int)cfg.n_heads;
    int head_dim = d_model / n_heads;
    int factor = (int)cfg.subsampling_factor;
    int n_mels_cfg = (int)cfg.n_mels;
    float ln_eps = 1e-5f;
    int n_rot = head_dim;

    int pad = (factor - (T % factor)) % factor;
    int T_padded = T + pad;
    int Tp = T_padded / factor;

    std::vector<int32_t> positions(Tp);
    for (int i = 0; i < Tp; ++i) positions[i] = i;

    pk::ensure_weights_realized(ml);
    pk::GraphInputPool pool;

    // ---- Stage 1: norm1 output ----
    {
        std::vector<float> out;
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t mel_ne[2] = {T_padded, n_mels_cfg};
            std::vector<float>& mel_padded = pool.alloc_f32((size_t)n_mels_cfg * T_padded);
            for (int mm = 0; mm < n_mels_cfg; ++mm)
                for (int t = 0; t < T; ++t)
                    mel_padded[(size_t)mm * T_padded + t] = feats[(size_t)mm * T + t];
            ggml_tensor* mel_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mel_ne,
                                  mel_padded.data(), (size_t)n_mels_cfg * T_padded * sizeof(float));
            mel_t = ggml_cont(ctx, mel_t);
            mel_t = ggml_cont(ctx, ggml_transpose(ctx, mel_t));
            ggml_tensor* stacked = ggml_reshape_2d(ctx, mel_t, (int64_t)n_mels_cfg * factor, Tp);
            ggml_tensor* proj_w = pk::clone_weight(ctx, ml, "encoder.pre_encode.proj.weight");
            ggml_tensor* x = ggml_mul_mat(ctx, proj_w, stacked);

            // embed_norm
            {
                ggml_tensor* g = pk::clone_weight(ctx, ml, "encoder.embed_norm.weight");
                ggml_tensor* b = pk::clone_weight(ctx, ml, "encoder.embed_norm.bias");
                x = ggml_norm(ctx, x, ln_eps);
                x = ggml_mul(ctx, x, g);
                x = ggml_add(ctx, x, b);
            }

            // norm1
            std::string base = "encoder.layers.0.";
            ggml_tensor* ng = pk::clone_weight(ctx, ml, (base + "norm1.weight").c_str());
            ggml_tensor* nb = pk::clone_weight(ctx, ml, (base + "norm1.bias").c_str());
            ggml_tensor* h = ggml_norm(ctx, x, ln_eps);
            h = ggml_mul(ctx, h, ng);
            h = ggml_add(ctx, h, nb);
            // h is [d_model, Tp] in ggml. PyTorch norm1 is [Tp, d_model].
            // ggml flat: [t * d_model + d] = same as PyTorch [t, d] row-major.
            return h;
        }, out);
        assert(ok);
        write_npy_f32(std::string(out_dir) + "/l0_norm1_pk.npy", out.data(),
                      {(int64_t)Tp, (int64_t)d_model});
        printf("norm1: [Tp=%d, d=%d] first5: ", Tp, d_model);
        for (int i = 0; i < 5; ++i) printf("%.4f ", out[i]);
        printf("\n");
    }

    // ---- Stage 2: QKV ----
    {
        std::vector<float> out;
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t mel_ne[2] = {T_padded, n_mels_cfg};
            std::vector<float>& mel_padded = pool.alloc_f32((size_t)n_mels_cfg * T_padded);
            for (int mm = 0; mm < n_mels_cfg; ++mm)
                for (int t = 0; t < T; ++t)
                    mel_padded[(size_t)mm * T_padded + t] = feats[(size_t)mm * T + t];
            ggml_tensor* mel_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mel_ne,
                                  mel_padded.data(), (size_t)n_mels_cfg * T_padded * sizeof(float));
            mel_t = ggml_cont(ctx, mel_t);
            mel_t = ggml_cont(ctx, ggml_transpose(ctx, mel_t));
            ggml_tensor* stacked = ggml_reshape_2d(ctx, mel_t, (int64_t)n_mels_cfg * factor, Tp);
            ggml_tensor* proj_w = pk::clone_weight(ctx, ml, "encoder.pre_encode.proj.weight");
            ggml_tensor* x = ggml_mul_mat(ctx, proj_w, stacked);
            {
                ggml_tensor* g = pk::clone_weight(ctx, ml, "encoder.embed_norm.weight");
                ggml_tensor* b = pk::clone_weight(ctx, ml, "encoder.embed_norm.bias");
                x = ggml_norm(ctx, x, ln_eps);
                x = ggml_mul(ctx, x, g);
                x = ggml_add(ctx, x, b);
            }
            std::string base = "encoder.layers.0.";
            ggml_tensor* ng = pk::clone_weight(ctx, ml, (base + "norm1.weight").c_str());
            ggml_tensor* nb = pk::clone_weight(ctx, ml, (base + "norm1.bias").c_str());
            ggml_tensor* h = ggml_norm(ctx, x, ln_eps);
            h = ggml_mul(ctx, h, ng);
            h = ggml_add(ctx, h, nb);

            // QKV
            ggml_tensor* qkv_w = pk::clone_weight(ctx, ml, (base + "attn.w_qkv.weight").c_str());
            ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, h); // [3*d, Tp]
            qkv = ggml_cont(ctx, qkv);
            // PyTorch qkv: [B, T, 3*D] → flat[t * 3*D + j]
            // ggml: [3*d, Tp] → flat[t * 3*d + j] — same!
            return qkv;
        }, out);
        assert(ok);
        write_npy_f32(std::string(out_dir) + "/l0_qkv_pk.npy", out.data(),
                      {(int64_t)Tp, (int64_t)(3 * d_model)});
        printf("qkv: [Tp=%d, 3d=%d] first5: ", Tp, 3*d_model);
        for (int i = 0; i < 5; ++i) printf("%.4f ", out[i]);
        printf("\n");
    }

    // ---- Stage 3: Q pre-RoPE ----
    {
        std::vector<float> out;
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t mel_ne[2] = {T_padded, n_mels_cfg};
            std::vector<float>& mel_padded = pool.alloc_f32((size_t)n_mels_cfg * T_padded);
            for (int mm = 0; mm < n_mels_cfg; ++mm)
                for (int t = 0; t < T; ++t)
                    mel_padded[(size_t)mm * T_padded + t] = feats[(size_t)mm * T + t];
            ggml_tensor* mel_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mel_ne,
                                  mel_padded.data(), (size_t)n_mels_cfg * T_padded * sizeof(float));
            mel_t = ggml_cont(ctx, mel_t);
            mel_t = ggml_cont(ctx, ggml_transpose(ctx, mel_t));
            ggml_tensor* stacked = ggml_reshape_2d(ctx, mel_t, (int64_t)n_mels_cfg * factor, Tp);
            ggml_tensor* proj_w = pk::clone_weight(ctx, ml, "encoder.pre_encode.proj.weight");
            ggml_tensor* x = ggml_mul_mat(ctx, proj_w, stacked);
            {
                ggml_tensor* g = pk::clone_weight(ctx, ml, "encoder.embed_norm.weight");
                ggml_tensor* b = pk::clone_weight(ctx, ml, "encoder.embed_norm.bias");
                x = ggml_norm(ctx, x, ln_eps);
                x = ggml_mul(ctx, x, g);
                x = ggml_add(ctx, x, b);
            }
            std::string base = "encoder.layers.0.";
            ggml_tensor* ng = pk::clone_weight(ctx, ml, (base + "norm1.weight").c_str());
            ggml_tensor* nb = pk::clone_weight(ctx, ml, (base + "norm1.bias").c_str());
            ggml_tensor* h = ggml_norm(ctx, x, ln_eps);
            h = ggml_mul(ctx, h, ng);
            h = ggml_add(ctx, h, nb);

            ggml_tensor* qkv_w = pk::clone_weight(ctx, ml, (base + "attn.w_qkv.weight").c_str());
            ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, h);
            qkv = ggml_cont(ctx, qkv);
            qkv = ggml_reshape_3d(ctx, qkv, d_model, 3, Tp);
            size_t qkv_ts = (size_t)3 * d_model * sizeof(float);
            ggml_tensor* q = ggml_view_2d(ctx, qkv, d_model, Tp, qkv_ts, 0);
            q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), head_dim, n_heads, Tp);
            // q: [hd, H, Tp] in ggml
            // PyTorch q: [H, T, hd] — flat[t * H * hd + h * hd + d] = [t * d_model + h * hd + d]
            // ggml [hd, H, Tp]: flat[t * H * hd + h * hd + d] — same!
            return q;
        }, out);
        assert(ok);
        write_npy_f32(std::string(out_dir) + "/l0_q_pk.npy", out.data(),
                      {(int64_t)Tp, (int64_t)d_model});
        printf("q_pre_rope: [Tp=%d, d=%d] first5: ", Tp, d_model);
        for (int i = 0; i < 5; ++i) printf("%.4f ", out[i]);
        printf("\n");
    }

    // ---- Stage 4: Q post-RoPE ----
    {
        std::vector<float> out;
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t mel_ne[2] = {T_padded, n_mels_cfg};
            std::vector<float>& mel_padded = pool.alloc_f32((size_t)n_mels_cfg * T_padded);
            for (int mm = 0; mm < n_mels_cfg; ++mm)
                for (int t = 0; t < T; ++t)
                    mel_padded[(size_t)mm * T_padded + t] = feats[(size_t)mm * T + t];
            ggml_tensor* mel_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mel_ne,
                                  mel_padded.data(), (size_t)n_mels_cfg * T_padded * sizeof(float));
            mel_t = ggml_cont(ctx, mel_t);
            mel_t = ggml_cont(ctx, ggml_transpose(ctx, mel_t));
            ggml_tensor* stacked = ggml_reshape_2d(ctx, mel_t, (int64_t)n_mels_cfg * factor, Tp);
            ggml_tensor* proj_w = pk::clone_weight(ctx, ml, "encoder.pre_encode.proj.weight");
            ggml_tensor* x = ggml_mul_mat(ctx, proj_w, stacked);
            {
                ggml_tensor* g = pk::clone_weight(ctx, ml, "encoder.embed_norm.weight");
                ggml_tensor* b = pk::clone_weight(ctx, ml, "encoder.embed_norm.bias");
                x = ggml_norm(ctx, x, ln_eps);
                x = ggml_mul(ctx, x, g);
                x = ggml_add(ctx, x, b);
            }
            std::string base = "encoder.layers.0.";
            ggml_tensor* ng = pk::clone_weight(ctx, ml, (base + "norm1.weight").c_str());
            ggml_tensor* nb = pk::clone_weight(ctx, ml, (base + "norm1.bias").c_str());
            ggml_tensor* h = ggml_norm(ctx, x, ln_eps);
            h = ggml_mul(ctx, h, ng);
            h = ggml_add(ctx, h, nb);

            int64_t pos_ne[1] = {Tp};
            ggml_tensor* pos = pk::graph_input_tensor(ctx, GGML_TYPE_I32, 1, pos_ne,
                                  positions.data(), (size_t)Tp * sizeof(int32_t));

            ggml_tensor* qkv_w = pk::clone_weight(ctx, ml, (base + "attn.w_qkv.weight").c_str());
            ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, h);
            qkv = ggml_cont(ctx, qkv);
            qkv = ggml_reshape_3d(ctx, qkv, d_model, 3, Tp);
            size_t qkv_ts = (size_t)3 * d_model * sizeof(float);
            ggml_tensor* q = ggml_view_2d(ctx, qkv, d_model, Tp, qkv_ts, 0);
            q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), head_dim, n_heads, Tp);
            q = ggml_rope_ext(ctx, q, pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, 0,
                    10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            return q;
        }, out);
        assert(ok);
        write_npy_f32(std::string(out_dir) + "/l0_q_rot_pk.npy", out.data(),
                      {(int64_t)Tp, (int64_t)d_model});
        printf("q_post_rope: [Tp=%d, d=%d] first5: ", Tp, d_model);
        for (int i = 0; i < 5; ++i) printf("%.4f ", out[i]);
        printf("\n");
    }

    // ---- Stage 5: attention output (pre out_proj) ----
    {
        std::vector<float> out;
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t mel_ne[2] = {T_padded, n_mels_cfg};
            std::vector<float>& mel_padded = pool.alloc_f32((size_t)n_mels_cfg * T_padded);
            for (int mm = 0; mm < n_mels_cfg; ++mm)
                for (int t = 0; t < T; ++t)
                    mel_padded[(size_t)mm * T_padded + t] = feats[(size_t)mm * T + t];
            ggml_tensor* mel_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mel_ne,
                                  mel_padded.data(), (size_t)n_mels_cfg * T_padded * sizeof(float));
            mel_t = ggml_cont(ctx, mel_t);
            mel_t = ggml_cont(ctx, ggml_transpose(ctx, mel_t));
            ggml_tensor* stacked = ggml_reshape_2d(ctx, mel_t, (int64_t)n_mels_cfg * factor, Tp);
            ggml_tensor* proj_w = pk::clone_weight(ctx, ml, "encoder.pre_encode.proj.weight");
            ggml_tensor* x = ggml_mul_mat(ctx, proj_w, stacked);
            {
                ggml_tensor* g = pk::clone_weight(ctx, ml, "encoder.embed_norm.weight");
                ggml_tensor* b = pk::clone_weight(ctx, ml, "encoder.embed_norm.bias");
                x = ggml_norm(ctx, x, ln_eps);
                x = ggml_mul(ctx, x, g);
                x = ggml_add(ctx, x, b);
            }
            std::string base = "encoder.layers.0.";
            ggml_tensor* ng = pk::clone_weight(ctx, ml, (base + "norm1.weight").c_str());
            ggml_tensor* nb = pk::clone_weight(ctx, ml, (base + "norm1.bias").c_str());
            ggml_tensor* h = ggml_norm(ctx, x, ln_eps);
            h = ggml_mul(ctx, h, ng);
            h = ggml_add(ctx, h, nb);

            int64_t pos_ne[1] = {Tp};
            ggml_tensor* pos = pk::graph_input_tensor(ctx, GGML_TYPE_I32, 1, pos_ne,
                                  positions.data(), (size_t)Tp * sizeof(int32_t));

            ggml_tensor* qkv_w = pk::clone_weight(ctx, ml, (base + "attn.w_qkv.weight").c_str());
            ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, h);
            qkv = ggml_cont(ctx, qkv);
            qkv = ggml_reshape_3d(ctx, qkv, d_model, 3, Tp);
            size_t qkv_ts = (size_t)3 * d_model * sizeof(float);
            ggml_tensor* q = ggml_view_2d(ctx, qkv, d_model, Tp, qkv_ts, 0);
            ggml_tensor* k = ggml_view_2d(ctx, qkv, d_model, Tp, qkv_ts, (size_t)d_model * sizeof(float));
            ggml_tensor* v = ggml_view_2d(ctx, qkv, d_model, Tp, qkv_ts, (size_t)2 * d_model * sizeof(float));
            q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), head_dim, n_heads, Tp);
            k = ggml_reshape_3d(ctx, ggml_cont(ctx, k), head_dim, n_heads, Tp);
            v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), head_dim, n_heads, Tp);
            q = ggml_rope_ext(ctx, q, pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, 0,
                    10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(ctx, k, pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, 0,
                    10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // Manual attention (same as test_diar_layer0.cpp)
            q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [hd, Tp, H]
            k = ggml_permute(ctx, k, 0, 2, 1, 3);
            v = ggml_permute(ctx, v, 0, 2, 1, 3);
            ggml_tensor* scores = ggml_mul_mat(ctx, k, q);  // [Tp, Tp, H]
            float attn_scale = 1.0f / std::sqrt((float)head_dim);
            scores = ggml_scale(ctx, scores, attn_scale);
            scores = ggml_soft_max(ctx, scores);
            ggml_tensor* v_t = ggml_permute(ctx, v, 1, 0, 2, 3);  // [Tp, hd, H]
            v_t = ggml_cont(ctx, v_t);
            ggml_tensor* attn_out = ggml_mul_mat(ctx, v_t, scores);  // [Tp, hd, H] -> [hd, Tp, H]?
            // Actually mul_mat(v_t [Tp,hd,H], scores [Tp,Tp,H]) gives [hd, Tp, H]
            attn_out = ggml_permute(ctx, attn_out, 1, 0, 2, 3);  // hmm

            // Let's just use flash_attn_ext instead
            (void)attn_out; // unused
            float scale = 1.0f / std::sqrt((float)head_dim);
            ggml_tensor* attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
            attn = ggml_permute(ctx, attn, 0, 2, 1, 3);  // [hd, H, Tp, 1]
            attn = ggml_cont(ctx, attn);
            attn = ggml_reshape_2d(ctx, attn, (int64_t)d_model, (int64_t)Tp);
            // attn: [d_model, Tp], flat[t * d + d] = same as PyTorch [Tp, d_model]
            return attn;
        }, out);
        assert(ok);
        write_npy_f32(std::string(out_dir) + "/l0_attn_pk.npy", out.data(),
                      {(int64_t)Tp, (int64_t)d_model});
        printf("attn: [Tp=%d, d=%d] first5: ", Tp, d_model);
        for (int i = 0; i < 5; ++i) printf("%.4f ", out[i]);
        printf("\n");
    }

    // ---- Stage 6: attn_out (after out_proj) ----
    {
        std::vector<float> out;
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t mel_ne[2] = {T_padded, n_mels_cfg};
            std::vector<float>& mel_padded = pool.alloc_f32((size_t)n_mels_cfg * T_padded);
            for (int mm = 0; mm < n_mels_cfg; ++mm)
                for (int t = 0; t < T; ++t)
                    mel_padded[(size_t)mm * T_padded + t] = feats[(size_t)mm * T + t];
            ggml_tensor* mel_t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, mel_ne,
                                  mel_padded.data(), (size_t)n_mels_cfg * T_padded * sizeof(float));
            mel_t = ggml_cont(ctx, mel_t);
            mel_t = ggml_cont(ctx, ggml_transpose(ctx, mel_t));
            ggml_tensor* stacked = ggml_reshape_2d(ctx, mel_t, (int64_t)n_mels_cfg * factor, Tp);
            ggml_tensor* proj_w = pk::clone_weight(ctx, ml, "encoder.pre_encode.proj.weight");
            ggml_tensor* x = ggml_mul_mat(ctx, proj_w, stacked);
            {
                ggml_tensor* g = pk::clone_weight(ctx, ml, "encoder.embed_norm.weight");
                ggml_tensor* b = pk::clone_weight(ctx, ml, "encoder.embed_norm.bias");
                x = ggml_norm(ctx, x, ln_eps);
                x = ggml_mul(ctx, x, g);
                x = ggml_add(ctx, x, b);
            }
            std::string base = "encoder.layers.0.";
            ggml_tensor* ng = pk::clone_weight(ctx, ml, (base + "norm1.weight").c_str());
            ggml_tensor* nb = pk::clone_weight(ctx, ml, (base + "norm1.bias").c_str());
            ggml_tensor* h = ggml_norm(ctx, x, ln_eps);
            h = ggml_mul(ctx, h, ng);
            h = ggml_add(ctx, h, nb);

            int64_t pos_ne[1] = {Tp};
            ggml_tensor* pos = pk::graph_input_tensor(ctx, GGML_TYPE_I32, 1, pos_ne,
                                  positions.data(), (size_t)Tp * sizeof(int32_t));

            ggml_tensor* qkv_w = pk::clone_weight(ctx, ml, (base + "attn.w_qkv.weight").c_str());
            ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, h);
            qkv = ggml_cont(ctx, qkv);
            qkv = ggml_reshape_3d(ctx, qkv, d_model, 3, Tp);
            size_t qkv_ts = (size_t)3 * d_model * sizeof(float);
            ggml_tensor* q = ggml_view_2d(ctx, qkv, d_model, Tp, qkv_ts, 0);
            ggml_tensor* k = ggml_view_2d(ctx, qkv, d_model, Tp, qkv_ts, (size_t)d_model * sizeof(float));
            ggml_tensor* v = ggml_view_2d(ctx, qkv, d_model, Tp, qkv_ts, (size_t)2 * d_model * sizeof(float));
            q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), head_dim, n_heads, Tp);
            k = ggml_reshape_3d(ctx, ggml_cont(ctx, k), head_dim, n_heads, Tp);
            v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), head_dim, n_heads, Tp);
            q = ggml_rope_ext(ctx, q, pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, 0,
                    10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(ctx, k, pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, 0,
                    10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            q = ggml_permute(ctx, q, 0, 2, 1, 3);
            k = ggml_permute(ctx, k, 0, 2, 1, 3);
            v = ggml_permute(ctx, v, 0, 2, 1, 3);
            float scale = 1.0f / std::sqrt((float)head_dim);
            ggml_tensor* attn = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
            attn = ggml_permute(ctx, attn, 0, 2, 1, 3);
            attn = ggml_cont(ctx, attn);
            attn = ggml_reshape_2d(ctx, attn, (int64_t)d_model, (int64_t)Tp);
            ggml_tensor* op_w = pk::clone_weight(ctx, ml, (base + "attn.out_proj.weight").c_str());
            attn = ggml_mul_mat(ctx, op_w, attn);
            ggml_tensor* op_b = pk::clone_weight_opt(ctx, ml, (base + "attn.out_proj.bias").c_str());
            if (op_b) attn = ggml_add(ctx, attn, op_b);
            return attn;
        }, out);
        assert(ok);
        write_npy_f32(std::string(out_dir) + "/l0_attn_out_pk.npy", out.data(),
                      {(int64_t)Tp, (int64_t)d_model});
        printf("attn_out: [Tp=%d, d=%d] first5: ", Tp, d_model);
        for (int i = 0; i < 5; ++i) printf("%.4f ", out[i]);
        printf("\n");
    }

    return 0;
}
