// test_diar_layer0.cpp — dump intermediate outputs after embed_norm and layer 0
// for comparison with the PyTorch reference.
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

    // Now build the graph manually up to embed_norm + layer 0
    const auto& cfg = ml.config();
    int d_model = (int)cfg.d_model;
    int n_heads = (int)cfg.n_heads;
    int head_dim = d_model / n_heads;
    int n_layers = (int)cfg.n_layers;
    int ff_dim = (int)cfg.ff_dim;
    int factor = (int)cfg.subsampling_factor;
    int n_mels_cfg = (int)cfg.n_mels;
    float ln_eps = 1e-5f;
    int n_rot = head_dim; // rotary_fraction=1.0

    int pad = (factor - (T % factor)) % factor;
    int T_padded = T + pad;
    int Tp = T_padded / factor;

    std::vector<int32_t> positions(Tp);
    for (int i = 0; i < Tp; ++i) positions[i] = i;

    pk::ensure_weights_realized(ml);
    pk::GraphInputPool pool;

    // Output: after embed_norm [Tp, d_model] (channels-last, like PyTorch)
    std::vector<float> embed_norm_out;
    bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
        // FeatureStacking
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
        ggml_tensor* x = ggml_mul_mat(ctx, proj_w, stacked); // [d_model, Tp]

        // embed_norm
        ggml_tensor* g = pk::clone_weight(ctx, ml, "encoder.embed_norm.weight");
        ggml_tensor* b = pk::clone_weight(ctx, ml, "encoder.embed_norm.bias");
        x = ggml_norm(ctx, x, ln_eps);
        x = ggml_mul(ctx, x, g);
        x = ggml_add(ctx, x, b);
        // x is [d_model, Tp] (channels-first in ggml)
        // PyTorch has [Tp, d_model] (channels-last)
        // Transpose for comparison
        x = ggml_cont(ctx, ggml_transpose(ctx, x));
        return x;
    }, embed_norm_out);
    assert(ok);
    write_npy_f32(std::string(out_dir) + "/embed_norm_pk.npy",
                  embed_norm_out.data(), {(int64_t)Tp, (int64_t)d_model});
    std::printf("embed_norm: [%d, %d]\n", Tp, d_model);
    std::printf("  first 5: ");
    for (int i = 0; i < 5; ++i) std::printf("%.4f ", embed_norm_out[i]);
    std::printf("\n");

    // Layer 0 full
    std::vector<float> layer0_out;
    ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
        // FeatureStacking + embed_norm
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
        // x: [d_model, Tp]

        // Position tensor
        int64_t pos_ne[1] = {Tp};
        ggml_tensor* pos = pk::graph_input_tensor(ctx, GGML_TYPE_I32, 1, pos_ne,
                              positions.data(), (size_t)Tp * sizeof(int32_t));

        // Layer 0
        std::string base = "encoder.layers.0.";

        // norm1
        {
            ggml_tensor* ng = pk::clone_weight(ctx, ml, (base + "norm1.weight").c_str());
            ggml_tensor* nb = pk::clone_weight(ctx, ml, (base + "norm1.bias").c_str());
            ggml_tensor* h = ggml_norm(ctx, x, ln_eps);
            h = ggml_mul(ctx, h, ng);
            h = ggml_add(ctx, h, nb);

            // QKV
            ggml_tensor* qkv_w = pk::clone_weight(ctx, ml, (base + "attn.w_qkv.weight").c_str());
            ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, h); // [1536, Tp]
            qkv = ggml_cont(ctx, qkv);
            qkv = ggml_reshape_3d(ctx, qkv, d_model, 3, Tp); // [512, 3, Tp]

            size_t qkv_ts = (size_t)3 * d_model * sizeof(float);
            ggml_tensor* q = ggml_view_2d(ctx, qkv, d_model, Tp, qkv_ts, 0);
            ggml_tensor* k = ggml_view_2d(ctx, qkv, d_model, Tp, qkv_ts, (size_t)d_model * sizeof(float));
            ggml_tensor* v = ggml_view_2d(ctx, qkv, d_model, Tp, qkv_ts, (size_t)2 * d_model * sizeof(float));

            q = ggml_reshape_3d(ctx, ggml_cont(ctx, q), head_dim, n_heads, Tp);
            k = ggml_reshape_3d(ctx, ggml_cont(ctx, k), head_dim, n_heads, Tp);
            v = ggml_reshape_3d(ctx, ggml_cont(ctx, v), head_dim, n_heads, Tp);

            // RoPE
            q = ggml_rope_ext(ctx, q, pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, 0,
                    10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(ctx, k, pos, nullptr, n_rot, GGML_ROPE_TYPE_NEOX, 0,
                    10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // Manual attention: permute to [hd, Tp, H]
            q = ggml_permute(ctx, q, 0, 2, 1, 3);
            k = ggml_permute(ctx, k, 0, 2, 1, 3);
            v = ggml_permute(ctx, v, 0, 2, 1, 3);

            // scores = mul_mat(k, q) → [Tp, Tp, H]
            ggml_tensor* scores = ggml_mul_mat(ctx, k, q);
            float attn_scale = 1.0f / std::sqrt((float)head_dim);
            scores = ggml_scale(ctx, scores, attn_scale);
            scores = ggml_soft_max(ctx, scores);

            // out = mul_mat(v_t, scores)
            ggml_tensor* v_t = ggml_permute(ctx, v, 1, 0, 2, 3); // [Tp, hd, H]
            v_t = ggml_cont(ctx, v_t);
            ggml_tensor* attn_out = ggml_mul_mat(ctx, v_t, scores); // [Tp, hd, H]
            attn_out = ggml_permute(ctx, attn_out, 1, 0, 2, 3); // [hd, Tp, H]
            attn_out = ggml_cont(ctx, attn_out);
            ggml_tensor* attn = ggml_reshape_2d(ctx, attn_out, (int64_t)d_model, (int64_t)Tp);

            // out_proj
            ggml_tensor* op_w = pk::clone_weight(ctx, ml, (base + "attn.out_proj.weight").c_str());
            attn = ggml_mul_mat(ctx, op_w, attn);
            ggml_tensor* op_b = pk::clone_weight_opt(ctx, ml, (base + "attn.out_proj.bias").c_str());
            if (op_b) attn = ggml_add(ctx, attn, op_b);

            // Residual
            x = ggml_add(ctx, x, attn);
        }

        // FFN
        {
            ggml_tensor* ng = pk::clone_weight(ctx, ml, (base + "norm2.weight").c_str());
            ggml_tensor* nb = pk::clone_weight(ctx, ml, (base + "norm2.bias").c_str());
            ggml_tensor* h = ggml_norm(ctx, x, ln_eps);
            h = ggml_mul(ctx, h, ng);
            h = ggml_add(ctx, h, nb);

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

        // x is [d_model, Tp] (ne[0]=d_model, ne[1]=Tp)
        // The output flat buffer is time-major: [d0_t0, d1_t0, ..., d511_t0, d0_t1, ...]
        // = flat[t * d_model + d]
        // PyTorch has [Tp, d_model] = flat[t * d_model + d] — same!
        // So NO transpose needed. Just return x directly.
        return x;
    }, layer0_out);
    assert(ok);
    write_npy_f32(std::string(out_dir) + "/layer0_pk.npy",
                  layer0_out.data(), {(int64_t)Tp, (int64_t)d_model});
    std::printf("layer0: [%d, %d]\n", Tp, d_model);
    std::printf("  first 5: ");
    for (int i = 0; i < 5; ++i) std::printf("%.4f ", layer0_out[i]);
    std::printf("\n");

    return 0;
}
