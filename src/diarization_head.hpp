#pragma once
#include "model_loader.hpp"
#include <vector>

namespace pk {

// Diarization head — NeMo SortformerModules forward_speaker_logits + upsample.
//
// Architecture (offline path, transformer_encoder is None for Nemotron-3):
//   enc_out [d_model, T_enc]  (channels-first, from FastConformer encoder)
//     → encoder_proj: Linear(d_model → tf_d_model)     → [tf_d_model, T_enc]
//     → subpixel_upsample: Conv1d(tf_d_model → tf_d_model*upsample, k=3, pad=1)
//         → reshape → [tf_d_model, T_enc * upsample]
//     → relu → first_hidden_to_hidden: Linear(tf_d_model → tf_d_model)
//     → relu → single_hidden_to_spks: Linear(tf_d_model → n_speakers)
//     → sigmoid
//         → probs [n_speakers, T_enc * upsample]
//
// Weight names (verbatim from state dict):
//   sortformer_modules.encoder_proj.{weight,bias}
//   sortformer_modules.subpixel_upsample.{weight,bias}
//   sortformer_modules.first_hidden_to_hidden.{weight,bias}
//   sortformer_modules.single_hidden_to_spks.{weight,bias}
class DiarizationHead {
public:
    explicit DiarizationHead(const ModelLoader& ml);

    // enc_out:  row-major [d_model, T_enc] — enc_out[c*T_enc + t]  (channels-first)
    // probs:    row-major [n_speakers, T_out] — probs[s*T_out + t]  (post-sigmoid)
    void forward(const std::vector<float>& enc_out, int d_model, int T_enc,
                 std::vector<float>& probs, int& n_spk, int& T_out) const;

    // Compute the per-frame probabilities for a sub-range of the encoder output.
    // Same as forward() but operates on enc_out[start_enc .. start_enc+count_enc-1].
    // Used by the streaming path to get probs for just the chunk portion.
    void forward_range(const std::vector<float>& enc_out, int d_model, int T_enc_total,
                       int start_enc, int count_enc,
                       std::vector<float>& probs, int& n_spk, int& T_out) const;

private:
    const ModelLoader& ml_;
    int d_model_;        // encoder d_model (512)
    int tf_d_model_;     // sortformer hidden (192)
    int n_spk_;          // number of speakers (8)
    int upsample_;       // upsample factor (8)
};

} // namespace pk
