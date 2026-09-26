#pragma once
#include "model_loader.hpp"
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace pk {

// DiarizationEncoder — pre-LN RoPE Transformer encoder for Nemotron-3-Diarization.
//
// This is NOT the FastConformer encoder used by ASR. Nemotron-3-Diarization uses
// a TransformerEncoder (not ConformerEncoder) with:
//   - FeatureStacking subsampling (8× stack → Linear, no bias)
//   - Pre-block LayerNorm (embed_norm)
//   - N × TransformerBlock (pre-norm): x = x + attn(norm1(x)); x = x + ffn(norm2(x))
//   - MultiHeadAttention: fused QKV (no bias) → RoPE → flash_attn → out_proj (bias)
//   - FeedForward: Linear → GELU → Linear (both with bias)
//   - Post-block LayerNorm (final_norm)
//   - RoPE: GPT-NeoX convention (GGML_ROPE_TYPE_NEOX), theta=10000, rotary_fraction=1.0
//
// Input:  mel features [n_mels, T] (row-major: mel[m*T + t])
// Output: enc_out [d_model, T_enc] (row-major: enc_out[c*T_enc + t], channels-first)
class DiarizationEncoder {
public:
    explicit DiarizationEncoder(const ModelLoader& ml);

    // mel:  row-major [n_mels, T] — mel[m*T + t]
    // enc_out: row-major [d_model, T_enc] — enc_out[c*T_enc + t] (channels-first)
    void forward(const std::vector<float>& mel, int n_mels, int T,
                 std::vector<float>& enc_out, int& d_model, int& T_enc) const;

    // --- Streaming split: pre_encode + transformer_forward ---

    // Pre-encoder: FeatureStacking + Linear(1024→512) + embed_norm.
    // mel:  row-major [n_mels, T] — mel[m*T + t]
    // emb:  row-major [d_model, T_enc] — emb[c*T_enc + t] (channels-first)
    // T_enc = T_padded / subsampling_factor
    void pre_encode(const std::vector<float>& mel, int n_mels, int T,
                    std::vector<float>& emb, int& d_model, int& T_enc) const;

    // Transformer blocks + final_norm (the second half of the encoder).
    // emb:    row-major [d_model, T_enc] — emb[c*T_enc + t] (channels-first)
    // enc_out: row-major [d_model, T_enc] — enc_out[c*T_enc + t]
    void transformer_forward(const std::vector<float>& emb, int d_model, int T_enc,
                             std::vector<float>& enc_out) const;

    int subsampling() const { return subsampling_factor_; }
    int n_mels() const { return n_mels_; }
    int d_model() const { return d_model_; }
    int n_layers() const { return n_layers_; }
    int n_heads() const { return n_heads_; }
    int head_dim() const { return head_dim_; }

private:
    const ModelLoader& ml_;
    int d_model_;        // encoder d_model (512)
    int n_layers_;       // number of transformer blocks (31)
    int n_heads_;        // attention heads (8)
    int head_dim_;       // d_model / n_heads (64)
    int ff_dim_;         // feed-forward inner dim (2048)
    int subsampling_factor_; // FeatureStacking factor (8)
    int n_mels_;         // mel features (128)
    bool qkv_bias_;      // QKV projection bias (false)
    bool pre_block_norm_; // apply embed_norm before blocks (true)
    float rope_base_;    // RoPE theta (10000.0)
    float rotary_fraction_; // fraction of head_dim rotated (1.0)
    float ln_eps_;       // LayerNorm epsilon (1e-5)
};

} // namespace pk
