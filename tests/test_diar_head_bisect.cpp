// test_diar_head_bisect.cpp — dump diarization head intermediates stage by stage.
#include "diarization.hpp"
#include "mel.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "audio_io.hpp"
#include "diarization_encoder.hpp"
#include "diarization_head.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
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
    pk::DiarizationEncoder encoder(ml);

    std::vector<float> feats;
    int n_mels = 0, T = 0;
    mel.compute(audio.samples, feats, n_mels, T);

    std::vector<float> enc_out;
    int d_model = 0, T_enc = 0;
    encoder.forward(feats, n_mels, T, enc_out, d_model, T_enc);

    const auto& cfg = ml.config();
    int tf = (int)cfg.diarization.tf_d_model;
    int n_spk = (int)cfg.diarization.n_speakers;
    int up = (int)cfg.diarization.upsample_factor;
    int T_out = T_enc * up;

    pk::ensure_weights_realized(ml);

    // Stage 1: encoder_proj output
    {
        std::vector<float> out;
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t xt_ne[2] = {T_enc, d_model};
            ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, xt_ne,
                                  enc_out.data(), (size_t)d_model * T_enc * sizeof(float));
            ggml_tensor* ep_w = ml.tensor("sortformer_modules.encoder_proj.weight");
            ggml_tensor* W_proj = ggml_reshape_2d(ctx, ep_w, d_model, tf);
            ggml_tensor* xt_t = ggml_cont(ctx, ggml_transpose(ctx, xt));
            ggml_tensor* proj = ggml_mul_mat(ctx, W_proj, xt_t);
            ggml_tensor* ep_b = ml.tensor("sortformer_modules.encoder_proj.bias");
            if (ep_b) proj = ggml_add(ctx, proj, ep_b);
            // proj: ne[0]=tf, ne[1]=T_enc
            // Transpose to ne[0]=T_enc, ne[1]=tf to match PyTorch [T, tf]
            proj = ggml_cont(ctx, ggml_transpose(ctx, proj));
            return proj;
        }, out);
        assert(ok);
        write_npy_f32(std::string(out_dir) + "/proj_pk.npy", out.data(),
                      {(int64_t)T_enc, (int64_t)tf});
        printf("proj: [%d, %d] first3: %.4f %.4f %.4f\n", T_enc, tf, out[0], out[1], out[2]);
    }

    // Stage 2: conv output (pre-bias, pre-reshape)
    {
        std::vector<float> out;
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t xt_ne[2] = {T_enc, d_model};
            ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, xt_ne,
                                  enc_out.data(), (size_t)d_model * T_enc * sizeof(float));
            ggml_tensor* ep_w = ml.tensor("sortformer_modules.encoder_proj.weight");
            ggml_tensor* W_proj = ggml_reshape_2d(ctx, ep_w, d_model, tf);
            ggml_tensor* xt_t = ggml_cont(ctx, ggml_transpose(ctx, xt));
            ggml_tensor* proj = ggml_mul_mat(ctx, W_proj, xt_t);
            ggml_tensor* ep_b = ml.tensor("sortformer_modules.encoder_proj.bias");
            if (ep_b) proj = ggml_add(ctx, proj, ep_b);

            // conv input: ne[0]=T_enc, ne[1]=tf, ne[2]=1
            ggml_tensor* conv_in = ggml_cont(ctx, ggml_transpose(ctx, proj));
            conv_in = ggml_reshape_3d(ctx, conv_in, T_enc, tf, 1);

            ggml_tensor* spk_w = ml.tensor("sortformer_modules.subpixel_upsample.weight");
            spk_w = ggml_cast(ctx, spk_w, GGML_TYPE_F16);
            conv_in = ggml_cast(ctx, conv_in, GGML_TYPE_F16);

            ggml_tensor* conv_out = ggml_conv_1d(ctx, spk_w, conv_in, 1, 1, 1);
            // conv_out: ne[0]=T_enc, ne[1]=tf*up, ne[2]=1
            // Reshape to 2D and transpose to [tf*up, T_enc]
            conv_out = ggml_reshape_2d(ctx, conv_out, T_enc, tf * up);
            conv_out = ggml_cont(ctx, ggml_transpose(ctx, conv_out));
            // ne[0]=tf*up, ne[1]=T_enc
            ggml_tensor* spk_b = ml.tensor("sortformer_modules.subpixel_upsample.bias");
            if (spk_b) conv_out = ggml_add(ctx, conv_out, spk_b);
            // Transpose to [T_enc, tf*up] to match PyTorch [T, OC]
            conv_out = ggml_cont(ctx, ggml_transpose(ctx, conv_out));
            return conv_out;
        }, out);
        assert(ok);
        write_npy_f32(std::string(out_dir) + "/conv_pk.npy", out.data(),
                      {(int64_t)T_enc, (int64_t)(tf * up)});
        printf("conv: [%d, %d] first3: %.4f %.4f %.4f\n", T_enc, tf*up, out[0], out[1], out[2]);
    }

    // Stage 3: upsampled (after subpixel reshape)
    // Matches diarization_head.cpp: add bias directly (no transpose), then subpixel reshape
    {
        std::vector<float> out;
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t xt_ne[2] = {T_enc, d_model};
            ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, xt_ne,
                                  enc_out.data(), (size_t)d_model * T_enc * sizeof(float));
            ggml_tensor* ep_w = ml.tensor("sortformer_modules.encoder_proj.weight");
            ggml_tensor* W_proj = ggml_reshape_2d(ctx, ep_w, d_model, tf);
            ggml_tensor* xt_t = ggml_cont(ctx, ggml_transpose(ctx, xt));
            ggml_tensor* proj = ggml_mul_mat(ctx, W_proj, xt_t);
            ggml_tensor* ep_b = ml.tensor("sortformer_modules.encoder_proj.bias");
            if (ep_b) proj = ggml_add(ctx, proj, ep_b);

            ggml_tensor* conv_in = ggml_cont(ctx, ggml_transpose(ctx, proj));
            conv_in = ggml_reshape_3d(ctx, conv_in, T_enc, tf, 1);

            ggml_tensor* spk_w = ml.tensor("sortformer_modules.subpixel_upsample.weight");
            spk_w = ggml_cast(ctx, spk_w, GGML_TYPE_F16);
            conv_in = ggml_cast(ctx, conv_in, GGML_TYPE_F16);

            ggml_tensor* conv_out = ggml_conv_1d(ctx, spk_w, conv_in, 1, 1, 1);
            // conv_out: ne=[T_enc, tf*up, 1], data: flat[t + c*T_enc]
            conv_out = ggml_reshape_2d(ctx, conv_out, T_enc, tf * up);

            // Add bias directly (no transpose)
            ggml_tensor* spk_b = ml.tensor("sortformer_modules.subpixel_upsample.bias");
            if (spk_b) {
                ggml_tensor* spk_b_2d = ggml_reshape_2d(ctx, spk_b, 1, tf * up);
                conv_out = ggml_add(ctx, conv_out, spk_b_2d);
            }

            // Subpixel reshape (matches head code)
            ggml_tensor* upsampled = ggml_reshape_3d(ctx, conv_out, T_enc, up, tf);
            upsampled = ggml_reshape_2d(ctx, upsampled, T_enc * up, tf);
            upsampled = ggml_cont(ctx, ggml_transpose(ctx, upsampled));
            // ne[0]=tf, ne[1]=T_out
            // Transpose to [T_out, tf] to match PyTorch
            upsampled = ggml_cont(ctx, ggml_transpose(ctx, upsampled));
            return upsampled;
        }, out);
        assert(ok);
        write_npy_f32(std::string(out_dir) + "/up_pk.npy", out.data(),
                      {(int64_t)T_out, (int64_t)tf});
        printf("up: [%d, %d] first3: %.4f %.4f %.4f\n", T_out, tf, out[0], out[1], out[2]);
    }

    return 0;
}
