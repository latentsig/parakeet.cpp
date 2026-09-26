#include "diarization_head.hpp"
#include "ggml_graph.hpp"
#include "backend.hpp"
#include "ggml.h"
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

namespace pk {

DiarizationHead::DiarizationHead(const ModelLoader& ml) : ml_(ml) {
    const auto& cfg = ml.config();
    d_model_   = (int)cfg.d_model;
    tf_d_model_ = (int)cfg.diarization.tf_d_model;
    n_spk_     = (int)cfg.diarization.n_speakers;
    upsample_  = (int)cfg.diarization.upsample_factor;
}

void DiarizationHead::forward(const std::vector<float>& enc_out, int d_model, int T_enc,
                               std::vector<float>& probs, int& n_spk, int& T_out) const {
    assert(d_model == d_model_);
    assert((int)enc_out.size() == d_model * T_enc);

    n_spk = n_spk_;
    const int up = upsample_;
    T_out = T_enc * up;

    // Memory budget: encoder_proj [d_model, tf], subpixel conv [tf*up, tf, 3],
    // intermediate [tf*up, T_enc], speaker linears, output [n_spk, T_out].
    const size_t mem_bytes =
        (size_t)128 * 1024 * 1024 +
        (size_t)(d_model * tf_d_model_ + tf_d_model_ * up * tf_d_model_ * 3 +
                 tf_d_model_ * up * T_enc + tf_d_model_ * T_out +
                 n_spk_ * T_out) * sizeof(float) * 4;

    const ModelLoader& ml = ml_;
    const int tf = tf_d_model_;
    const int ns = n_spk_;

    pk::ensure_weights_realized(ml);

    bool ok = pk::run_graph(mem_bytes, /*n_threads=*/4,
        [&](ggml_context* ctx) -> ggml_tensor* {
            // ---- Input: enc_out [d_model, T_enc] row-major, enc_out[c*T_enc + t]
            // ggml: ne[0]=T_enc (fastest), ne[1]=d_model
            int64_t xt_ne[2] = {T_enc, d_model};
            ggml_tensor* xt = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, xt_ne,
                                  enc_out.data(), (size_t)d_model * T_enc * sizeof(float));

            // ---- encoder_proj: Linear(d_model → tf_d_model) ----
            // Weight: PyTorch [tf, d_model] → ggml ne[0]=d_model, ne[1]=tf
            ggml_tensor* ep_w = ml.tensor("sortformer_modules.encoder_proj.weight");
            if (!ep_w) throw std::runtime_error("parakeet: missing sortformer_modules.encoder_proj.weight");
            ggml_tensor* W_proj = ggml_reshape_2d(ctx, ep_w, d_model, tf);

            // Transpose xt so ne[0]=d_model (contraction dim)
            ggml_tensor* xt_t = ggml_cont(ctx, ggml_transpose(ctx, xt));
            // xt_t: ne[0]=d_model, ne[1]=T_enc

            ggml_tensor* proj = ggml_mul_mat(ctx, W_proj, xt_t);
            // proj: ne[0]=tf, ne[1]=T_enc

            // Add encoder_proj bias [tf]
            ggml_tensor* ep_b = ml.tensor("sortformer_modules.encoder_proj.bias");
            if (ep_b) proj = ggml_add(ctx, proj, ep_b);

            // ---- subpixel_upsample: Conv1d(tf → tf*up, k=3, pad=1) + bias ----
            // Conv1d expects data as ne[0]=T, ne[1]=IC, ne[2]=N
            // proj is ne[0]=tf, ne[1]=T_enc → need transpose
            ggml_tensor* conv_in = ggml_cont(ctx, ggml_transpose(ctx, proj));
            // conv_in: ne[0]=T_enc, ne[1]=tf
            conv_in = ggml_reshape_3d(ctx, conv_in, T_enc, tf, 1);
            // conv_in: ne[0]=T_enc, ne[1]=tf, ne[2]=1

            // Conv1d weight: GGUF stores ne=[k, IC, OC]=[3, 192, 1536] (converter
            // already wrote it in ggml layout). Use directly.
            ggml_tensor* spk_w = ml.tensor("sortformer_modules.subpixel_upsample.weight");
            if (!spk_w) throw std::runtime_error("parakeet: missing sortformer_modules.subpixel_upsample.weight");

            // ggml's CPU im2col expects the conv kernel to be F16 (assertion
            // in ggml_compute_forward_im2col_f16). Cast it explicitly.
            spk_w = ggml_cast(ctx, spk_w, GGML_TYPE_F16);
            // Also cast the input to F16 for the im2col path
            conv_in = ggml_cast(ctx, conv_in, GGML_TYPE_F16);

            // ggml_conv_1d(ctx, kernel, data, stride=1, pad=1, dilation=1)
            ggml_tensor* conv_out = ggml_conv_1d(ctx, spk_w, conv_in, 1, 1, 1);
            // conv_out: ne[0]=T_enc, ne[1]=tf*up, ne[2]=1
            // Data layout: flat[t + c*T_enc] (ne[0]=T_enc fastest)

            // Reshape to 2D (keep ne[0]=T_enc, ne[1]=tf*up)
            conv_out = ggml_reshape_2d(ctx, conv_out, T_enc, tf * up);
            // ne[0]=T_enc, ne[1]=tf*up, data: flat[t + c*T_enc]

            // Add subpixel bias [tf*up] directly. Reshape to [1, tf*up]
            // so ggml_add broadcasts over ne[0]=T_enc.
            ggml_tensor* spk_b = ml.tensor("sortformer_modules.subpixel_upsample.bias");
            if (spk_b) {
                ggml_tensor* spk_b_2d = ggml_reshape_2d(ctx, spk_b, 1, tf * up);
                conv_out = ggml_add(ctx, conv_out, spk_b_2d);
            }

            // Subpixel reshape: conv_out is ne=[T_enc, tf*up], data: flat[t + c*T_enc].
            //
            // Reference PyTorch: x.view(B, C//up, up, T) then x.view(B, C//up, up*T)
            //   → up_pk[h, t'] = conv[h*up+u, t] where t' = u*T + t
            //
            // In ggml (column-major, ne[0] fastest):
            // 1. reshape_3d(T_enc, up, tf): ne=[T_enc, up, tf]
            //    element(t,u,h) = flat[t + u*T_enc + h*up*T_enc] = flat[t + c*T_enc] ✓
            // 2. reshape_2d(T_out, tf): ne=[T_out, tf]
            //    element(t',h) = flat[t' + h*T_out] where t' = t + u*T_enc ✓
            // 3. transpose: ne=[tf, T_out]
            // 4. cont: copies to flat[h + t'*tf] (2D cont works correctly)
            //
            // NOTE: 3D permute+cont is BROKEN in this ggml backend — the cont op
            // does not actually rearrange data for 3D tensors. Using 2D
            // transpose+cont avoids this bug.
            ggml_tensor* upsampled = ggml_reshape_3d(ctx, conv_out, T_enc, up, tf);
            upsampled = ggml_reshape_2d(ctx, upsampled, T_enc * up, tf);
            upsampled = ggml_cont(ctx, ggml_transpose(ctx, upsampled));
            // ne[0]=tf, ne[1]=T_out

            // ---- forward_speaker_logits: relu → Linear(tf→tf) → relu → Linear(tf→ns) → sigmoid ----

            // First ReLU
            ggml_tensor* h = ggml_relu(ctx, upsampled);

            // first_hidden_to_hidden: Linear(tf → tf)
            ggml_tensor* fh_w = ml.tensor("sortformer_modules.first_hidden_to_hidden.weight");
            if (!fh_w) throw std::runtime_error("parakeet: missing sortformer_modules.first_hidden_to_hidden.weight");
            ggml_tensor* W1 = ggml_reshape_2d(ctx, fh_w, tf, tf);
            // W1: ne[0]=tf, ne[1]=tf. h: ne[0]=tf, ne[1]=T_out
            h = ggml_mul_mat(ctx, W1, h);
            // h: ne[0]=tf, ne[1]=T_out
            ggml_tensor* fh_b = ml.tensor("sortformer_modules.first_hidden_to_hidden.bias");
            if (fh_b) h = ggml_add(ctx, h, fh_b);

            // Second ReLU
            h = ggml_relu(ctx, h);

            // single_hidden_to_spks: Linear(tf → ns)
            ggml_tensor* ss_w = ml.tensor("sortformer_modules.single_hidden_to_spks.weight");
            if (!ss_w) throw std::runtime_error("parakeet: missing sortformer_modules.single_hidden_to_spks.weight");
            ggml_tensor* W2 = ggml_reshape_2d(ctx, ss_w, tf, ns);
            // W2: ne[0]=tf, ne[1]=ns. h: ne[0]=tf, ne[1]=T_out
            h = ggml_mul_mat(ctx, W2, h);
            // h: ne[0]=ns, ne[1]=T_out
            ggml_tensor* ss_b = ml.tensor("sortformer_modules.single_hidden_to_spks.bias");
            if (ss_b) h = ggml_add(ctx, h, ss_b);

            // Sigmoid → speaker probabilities
            ggml_tensor* sig = ggml_sigmoid(ctx, h);
            // sig: ne[0]=ns, ne[1]=T_out → memory: probs[t*ns + s]
            // postprocess expects row-major: probs[s*T_out + t]
            // Transpose to ne=[T_out, ns] + cont → flat[t + s*T_out]
            sig = ggml_cont(ctx, ggml_transpose(ctx, sig));

            return sig;
        },
        probs);

    assert(ok && "diarization_head graph failed");
    (void)ok;
}

} // namespace pk
