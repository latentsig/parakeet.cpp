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

            // ---- subpixel_upsample: Conv1d(tf -> tf*up, k=3, pad=1) + bias ----
            // im2col + mul_mat in F32 (ggml_conv_1d would force an F16 im2col).
            // im2col wants data as ne=[T, IC, N]; proj is ne=[tf, T_enc].
            ggml_tensor* conv_in = ggml_cont(ctx, ggml_transpose(ctx, proj));
            conv_in = ggml_reshape_3d(ctx, conv_in, T_enc, tf, 1);

            // GGUF stores the PyTorch [OC, IC, k] weight as ggml ne=[k, IC, OC].
            ggml_tensor* spk_w = ml.tensor("sortformer_modules.subpixel_upsample.weight");
            if (!spk_w) throw std::runtime_error("parakeet: missing sortformer_modules.subpixel_upsample.weight");
            ggml_tensor* cols = ggml_im2col(ctx, spk_w, conv_in, /*s0*/1, /*s1*/0,
                                            /*p0*/1, /*p1*/0, /*d0*/1, /*d1*/0,
                                            /*is_2D*/false, GGML_TYPE_F32);
            // cols: ne=[k*IC, T_enc, 1]
            cols = ggml_reshape_2d(ctx, cols, cols->ne[0], T_enc);
            ggml_tensor* w2d = spk_w->type == GGML_TYPE_F32
                ? spk_w : ggml_cast(ctx, spk_w, GGML_TYPE_F32);
            w2d = ggml_reshape_2d(ctx, w2d, spk_w->ne[0] * spk_w->ne[1], spk_w->ne[2]);
            ggml_tensor* conv_out = ggml_mul_mat(ctx, w2d, cols);
            // conv_out: ne=[OC=tf*up, T_enc], flat[c + t*OC]

            ggml_tensor* spk_b = ml.tensor("sortformer_modules.subpixel_upsample.bias");
            if (spk_b) conv_out = ggml_add(ctx, conv_out, spk_b);

            // Subpixel shuffle, NeMo SortformerModules.upsample_hidden:
            //   conv(x).transpose(1,2).reshape(B, T, up, tf).reshape(B, T*up, tf)
            // so output frame t*up+u, hidden h reads conv channel u*tf+h at
            // frame t. With conv_out time-major (flat[u*tf + h + t*tf*up]) this
            // is a plain reshape: element (h, t*up+u) = flat[h + (t*up+u)*tf].
            ggml_tensor* upsampled = ggml_reshape_2d(ctx, conv_out, tf, (int64_t)T_enc * up);
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

void DiarizationHead::forward_range(const std::vector<float>& enc_out, int d_model, int T_enc_total,
                                      int start_enc, int count_enc,
                                      std::vector<float>& probs, int& n_spk, int& T_out) const {
    assert(d_model == d_model_);
    assert(start_enc >= 0 && count_enc > 0 && start_enc + count_enc <= T_enc_total);

    // Extract the sub-range [start_enc, start_enc+count_enc) from enc_out.
    // enc_out is channels-first: enc_out[c*T_enc_total + t].
    std::vector<float> sub((size_t)d_model * count_enc);
    for (int c = 0; c < d_model; ++c)
        for (int t = 0; t < count_enc; ++t)
            sub[(size_t)c * count_enc + t] =
                enc_out[(size_t)c * T_enc_total + (start_enc + t)];

    forward(sub, d_model, count_enc, probs, n_spk, T_out);
}

} // namespace pk
