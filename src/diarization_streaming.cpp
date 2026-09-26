#include "diarization_streaming.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace pk {

StreamingDiarization::StreamingDiarization(const ModelLoader& ml)
    : ml_(ml)
    , encoder_(ml)
    , head_(ml)
{
    const auto& cfg = ml.config();
    const auto& d = cfg.diarization;

    d_model_            = (int)cfg.d_model;
    tf_d_model_         = (int)d.tf_d_model;
    n_spk_              = (int)d.n_speakers;
    n_layers_           = (int)cfg.n_layers;
    n_heads_            = (int)cfg.n_heads;
    head_dim_           = d_model_ / n_heads_;
    ff_dim_             = (int)cfg.ff_dim;
    subsampling_factor_ = encoder_.subsampling();
    upsample_factor_    = (int)d.upsample_factor;
    n_mels_             = encoder_.n_mels();
    chunk_len_          = d.chunk_len;
    spkcache_len_       = d.spkcache_len;
    fifo_len_           = d.fifo_len;
    frame_sec_          = d.frame_resolution_sec;
    onset_threshold_    = d.onset_threshold;
    offset_threshold_   = d.offset_threshold;

    // AOSC tuning
    spkcache_sil_frames_per_spk_ = d.spkcache_sil_frames_per_spk;
    sil_threshold_              = d.sil_threshold;
    pred_score_threshold_       = d.pred_score_threshold;
    scores_boost_latest_        = d.scores_boost_latest;
    strong_boost_rate_          = d.strong_boost_rate;
    weak_boost_rate_            = d.weak_boost_rate;
    min_pos_scores_rate_        = d.min_pos_scores_rate;
    max_index_                  = 99999;

    // Load the learned silence embedding if present
    if (d.use_learnable_sil_emb) {
        const ggml_tensor* t = ml.tensor("sortformer_modules.learnable_sil_emb");
        if (t) {
            has_silence_emb_ = true;
            silence_emb_.resize(d_model_);
            ensure_weights_realized(ml);
            const float* data = (const float*)t->data;
            std::memcpy(silence_emb_.data(), data, d_model_ * sizeof(float));
        }
    }

    mean_sil_emb_.assign(d_model_, 0.0f);
    n_sil_frames_ = 0;
    total_mel_frames_ = 0;
    spkcache_enc_len_ = 0;
    spkcache_preds_valid_ = false;
    fifo_enc_len_ = 0;
}

StreamingDiarization::~StreamingDiarization() = default;

void StreamingDiarization::reset() {
    spkcache_embs_.clear();
    spkcache_preds_.clear();
    spkcache_enc_len_ = 0;
    spkcache_preds_valid_ = false;
    fifo_embs_.clear();
    fifo_preds_.clear();
    fifo_enc_len_ = 0;
    mean_sil_emb_.assign(d_model_, 0.0f);
    n_sil_frames_ = 0;
    total_mel_frames_ = 0;
}

std::vector<StreamingSpeakerSegment> StreamingDiarization::feed_mel_chunk(
        const std::vector<float>& mel_chunk, int n_mels, int n_frames,
        bool is_last) {

    assert(n_mels == n_mels_);
    assert((int)mel_chunk.size() == n_mels * n_frames);

    const int factor = subsampling_factor_;  // 8
    // Pad the chunk to a multiple of subsampling_factor
    const int pad = (factor - (n_frames % factor)) % factor;
    const int T_padded = n_frames + pad;
    const int chunk_enc_len = T_padded / factor;

    // --- 1. Pre-encode the chunk ---
    std::vector<float> chunk_emb;
    int d_model = 0, T_enc_chunk = 0;
    encoder_.pre_encode(mel_chunk, n_mels, n_frames, chunk_emb, d_model, T_enc_chunk);
    assert(d_model == d_model_);
    assert(T_enc_chunk == chunk_enc_len);

    // --- 2. Concatenate [spkcache | chunk] for the encoder input ---
    // Build the combined mel input: [spkcache_mel | chunk_mel]
    // But wait — we work at the EMBEDDING level for spkcache, not mel level.
    // The spkcache stores pre-encoded embeddings. So we concatenate at the
    // embedding level: [spkcache_embs | chunk_emb], then run transformer_forward.

    int spk_enc = spkcache_enc_len_;
    int total_enc = spk_enc + chunk_enc_len;

    std::vector<float> combined_emb((size_t)d_model_ * total_enc);
    if (spk_enc > 0) {
        std::memcpy(combined_emb.data(), spkcache_embs_.data(),
                    (size_t)d_model_ * spk_enc * sizeof(float));
    }
    std::memcpy(combined_emb.data() + (size_t)d_model_ * spk_enc,
                chunk_emb.data(),
                (size_t)d_model_ * chunk_enc_len * sizeof(float));

    // --- 3. Run the transformer over [spkcache | chunk] ---
    std::vector<float> enc_out;
    encoder_.transformer_forward(combined_emb, d_model_, total_enc, enc_out);

    // --- 4. Run the diarization head on the CHUNK portion only ---
    // enc_out is [d_model, total_enc] channels-first.
    // The chunk portion starts at frame spk_enc.
    std::vector<float> probs;
    int n_spk = 0, T_out = 0;
    head_.forward_range(enc_out, d_model_, total_enc, spk_enc, chunk_enc_len,
                        probs, n_spk, T_out);
    assert(n_spk == n_spk_);

    // --- 5. Also get probs for the spkcache portion (for scoring) ---
    std::vector<float> full_probs;
    int full_n_spk = 0, full_T_out = 0;
    if (spk_enc > 0) {
        head_.forward_range(enc_out, d_model_, total_enc, 0, spk_enc,
                            full_probs, full_n_spk, full_T_out);
        assert(full_n_spk == n_spk_);
    }

    // --- 6. Post-process the chunk probs into segments ---
    float time_offset = total_mel_frames_ * frame_sec_;
    auto segments = postprocess_chunk(probs, n_spk_, T_out, time_offset);

    // --- 7. Update stream state (FIFO → spkcache → compress) ---
    if (!is_last) {
        // Build the full prediction output [n_spk, total_out] for the spkcache
        // portion + chunk portion, used by stream_state_update.
        int spk_out = 0;
        if (spk_enc > 0 && full_T_out > 0) {
            spk_out = full_T_out;
        }

        // The chunk portion of the prediction for scoring during compression
        stream_state_update(chunk_emb, chunk_enc_len,
                            probs, T_out,
                            full_probs, full_T_out);
    }

    total_mel_frames_ += n_frames;
    return segments;
}

void StreamingDiarization::boost_topk_scores(
        float* scores, int n_frames, int n_spk,
        int k_per_spk, float scale_factor, float offset) const {
    if (k_per_spk <= 0 || k_per_spk > n_frames) return;
    float boost = -scale_factor * std::log(offset);

    for (int s = 0; s < n_spk; ++s) {
        std::vector<std::pair<float, int>> sv(n_frames);
        for (int t = 0; t < n_frames; ++t) {
            sv[t] = {scores[(size_t)t * n_spk + s], t};
        }
        std::nth_element(sv.begin(), sv.begin() + k_per_spk, sv.end(),
            [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
                return a.first > b.first;
            });
        for (int i = 0; i < k_per_spk; ++i) {
            scores[(size_t)sv[i].second * n_spk + s] += boost;
        }
    }
}

void StreamingDiarization::update_silence_profile(
        const float* pop_embs, const float* pop_preds,
        int pop_len) {
    for (int t = 0; t < pop_len; ++t) {
        float pred_sum = 0;
        for (int s = 0; s < n_spk_; ++s) {
            pred_sum += pop_preds[(size_t)t * n_spk_ + s];
        }
        if (pred_sum < sil_threshold_) {
            ++n_sil_frames_;
            float w_old = (float)(n_sil_frames_ - 1) / (float)n_sil_frames_;
            float w_new = 1.0f / (float)n_sil_frames_;
            for (int d = 0; d < d_model_; ++d) {
                mean_sil_emb_[d] = w_old * mean_sil_emb_[d] +
                                   w_new * pop_embs[(size_t)t * d_model_ + d];
            }
        }
    }
}

void StreamingDiarization::compress_spkcache() {
    const int n_frames = spkcache_enc_len_;
    // Target spkcache size in ENCODER frames
    const int target_enc_len = spkcache_len_ / subsampling_factor_;
    const int sil_per_spk = spkcache_sil_frames_per_spk_;
    const int per_spk = target_enc_len / n_spk_ - sil_per_spk;
    const int strong_k = (int)std::floor(per_spk * strong_boost_rate_);
    const int weak_k   = (int)std::floor(per_spk * weak_boost_rate_);
    const int min_pos_k = (int)std::floor(per_spk * min_pos_scores_rate_);

    // 1. Compute log-based importance scores [n_frames, n_spk]
    std::vector<float> scores((size_t)n_frames * n_spk_);
    for (int t = 0; t < n_frames; ++t) {
        const float* p = &spkcache_preds_[(size_t)t * n_spk_];
        float log_1_sum = 0;
        for (int s = 0; s < n_spk_; ++s)
            log_1_sum += std::log(std::max(1.0f - p[s], pred_score_threshold_));
        for (int s = 0; s < n_spk_; ++s) {
            float lp  = std::log(std::max(p[s], pred_score_threshold_));
            float l1p = std::log(std::max(1.0f - p[s], pred_score_threshold_));
            scores[(size_t)t * n_spk_ + s] = lp - l1p + log_1_sum - std::log(0.5f);
        }
    }

    // 2. Disable non-speech scores (preds <= 0.5 → -inf)
    for (int t = 0; t < n_frames; ++t)
        for (int s = 0; s < n_spk_; ++s)
            if (spkcache_preds_[(size_t)t * n_spk_ + s] <= 0.5f)
                scores[(size_t)t * n_spk_ + s] = -INFINITY;

    // Disable non-positive scores if speaker has enough positive ones
    for (int s = 0; s < n_spk_; ++s) {
        int pos_cnt = 0;
        for (int t = 0; t < n_frames; ++t)
            if (scores[(size_t)t * n_spk_ + s] > 0) ++pos_cnt;
        if (pos_cnt >= min_pos_k) {
            for (int t = 0; t < n_frames; ++t) {
                if (scores[(size_t)t * n_spk_ + s] <= 0 &&
                    spkcache_preds_[(size_t)t * n_spk_ + s] > 0.5f)
                    scores[(size_t)t * n_spk_ + s] = -INFINITY;
            }
        }
    }

    // 3. Boost latest frames (beyond target_enc_len)
    if (scores_boost_latest_ > 0) {
        for (int t = target_enc_len; t < n_frames; ++t)
            for (int s = 0; s < n_spk_; ++s) {
                float& sc = scores[(size_t)t * n_spk_ + s];
                if (sc != -INFINITY) sc += scores_boost_latest_;
            }
    }

    // 4. Strong boost: top-K per speaker (scale=2)
    boost_topk_scores(scores.data(), n_frames, n_spk_, strong_k, 2.0f, 0.5f);

    // 5. Weak boost: top-K per speaker (scale=1)
    int wk = std::min(weak_k, n_frames);
    boost_topk_scores(scores.data(), n_frames, n_spk_, wk, 1.0f, 0.5f);

    // 6. Add silence placeholder frames at end (+inf for each speaker)
    int n_sil_pad = sil_per_spk;
    int n_total = n_frames + n_sil_pad;
    scores.resize((size_t)n_total * n_spk_);
    for (int t = n_frames; t < n_total; ++t)
        for (int s = 0; s < n_spk_; ++s)
            scores[(size_t)t * n_spk_ + s] = INFINITY;

    // 7. Flatten as (n_spk, n_total) and find top target_enc_len entries
    int flat_len = n_spk_ * n_total;
    std::vector<std::pair<float, int>> flat(flat_len);
    for (int s = 0; s < n_spk_; ++s)
        for (int t = 0; t < n_total; ++t)
            flat[(size_t)s * n_total + t] = {scores[(size_t)t * n_spk_ + s],
                                              s * n_total + t};

    std::nth_element(flat.begin(), flat.begin() + target_enc_len, flat.end(),
        [](const std::pair<float,int>& a, const std::pair<float,int>& b) {
            return a.first > b.first;
        });

    // Replace -inf entries with max_index, keep valid entries
    std::vector<int> topk_indices(target_enc_len);
    for (int i = 0; i < target_enc_len; ++i) {
        if (flat[i].first == -INFINITY) {
            topk_indices[i] = max_index_;
        } else {
            topk_indices[i] = flat[i].second;
        }
    }

    // Sort to preserve original frame order
    std::sort(topk_indices.begin(), topk_indices.end());

    // Convert to frame indices and determine disabled mask
    int n_frames_no_sil = n_total - n_sil_pad;
    std::vector<bool> is_disabled(target_enc_len, false);
    for (int i = 0; i < target_enc_len; ++i) {
        if (topk_indices[i] == max_index_) {
            is_disabled[i] = true;
        }
        topk_indices[i] = topk_indices[i] % n_total;
        if (topk_indices[i] >= n_frames_no_sil) {
            is_disabled[i] = true;
        }
        if (is_disabled[i]) {
            topk_indices[i] = 0;  // placeholder for gather
        }
    }

    // 8. Gather embeddings and predictions
    std::vector<float> new_embs((size_t)target_enc_len * d_model_);
    std::vector<float> new_preds((size_t)target_enc_len * n_spk_);

    // Use learned silence emb if available, otherwise running mean
    const float* sil_emb = has_silence_emb_ ? silence_emb_.data() : mean_sil_emb_.data();

    for (int i = 0; i < target_enc_len; ++i) {
        int tidx = topk_indices[i];
        if (is_disabled[i]) {
            std::memcpy(&new_embs[(size_t)i * d_model_], sil_emb, d_model_ * sizeof(float));
            std::memset(&new_preds[(size_t)i * n_spk_], 0, n_spk_ * sizeof(float));
        } else {
            std::memcpy(&new_embs[(size_t)i * d_model_],
                        &spkcache_embs_[(size_t)tidx * d_model_],
                        d_model_ * sizeof(float));
            std::memcpy(&new_preds[(size_t)i * n_spk_],
                        &spkcache_preds_[(size_t)tidx * n_spk_],
                        n_spk_ * sizeof(float));
        }
    }

    spkcache_embs_ = std::move(new_embs);
    spkcache_preds_ = std::move(new_preds);
    spkcache_enc_len_ = target_enc_len;
}

void StreamingDiarization::stream_state_update(
        const std::vector<float>& chunk_preenc, int chunk_enc_len,
        const std::vector<float>& chunk_preds, int chunk_out_len,
        const std::vector<float>& full_pred_out, int full_out_len) {

    int old_sc_len = spkcache_enc_len_;
    int old_fifo_len = fifo_enc_len_;

    // With fifo_len=0, the FIFO immediately overflows on every chunk.
    // The chunk embeddings are popped from the FIFO and appended to the spkcache.

    // Extract chunk predictions (chunk_preds is already [n_spk, chunk_out_len])
    // We need predictions in [chunk_enc_len, n_spk] format.
    // chunk_out_len should equal chunk_enc_len * upsample_factor, but the
    // predictions used for scoring are at encoder resolution, so we subsample.
    //
    // Actually, the spkcache stores ENCODER-resolution predictions.
    // The diarization head outputs at upsampled resolution (T_out = chunk_enc_len * upsample).
    // For spkcache scoring, NeMo uses the pre-sigmoid predictions at encoder resolution.
    // We approximate by taking the mean of each upsample_factor block.
    std::vector<float> chunk_preds_enc((size_t)chunk_enc_len * n_spk_, 0.0f);
    if (chunk_out_len == chunk_enc_len) {
        // Already at encoder resolution
        std::memcpy(chunk_preds_enc.data(), chunk_preds.data(),
                    chunk_preds.size() * sizeof(float));
    } else if (chunk_out_len > 0 && chunk_out_len % chunk_enc_len == 0) {
        int up = chunk_out_len / chunk_enc_len;
        for (int t = 0; t < chunk_enc_len; ++t) {
            for (int s = 0; s < n_spk_; ++s) {
                float sum = 0;
                for (int u = 0; u < up; ++u) {
                    int idx = t * up + u;
                    sum += chunk_preds[(size_t)s * chunk_out_len + idx];
                }
                chunk_preds_enc[(size_t)t * n_spk_ + s] = sum / up;
            }
        }
    }

    // Append chunk to FIFO
    int new_fifo_total = old_fifo_len + chunk_enc_len;
    std::vector<float> updated_fifo((size_t)(old_fifo_len + chunk_enc_len) * d_model_);
    std::vector<float> updated_fifo_preds((size_t)(old_fifo_len + chunk_enc_len) * n_spk_);

    if (old_fifo_len > 0) {
        std::memcpy(updated_fifo.data(), fifo_embs_.data(),
                    (size_t)old_fifo_len * d_model_ * sizeof(float));
        std::memcpy(updated_fifo_preds.data(), fifo_preds_.data(),
                    (size_t)old_fifo_len * n_spk_ * sizeof(float));
    }
    std::memcpy(updated_fifo.data() + (size_t)old_fifo_len * d_model_,
                chunk_preenc.data(),
                (size_t)chunk_enc_len * d_model_ * sizeof(float));
    std::memcpy(updated_fifo_preds.data() + (size_t)old_fifo_len * n_spk_,
                chunk_preds_enc.data(),
                (size_t)chunk_enc_len * n_spk_ * sizeof(float));

    // FIFO target in encoder frames
    int fifo_enc_target = fifo_len_ / subsampling_factor_;

    if (new_fifo_total > fifo_enc_target) {
        int pop_out_len = spkcache_len_ / subsampling_factor_;  // spkcache_update_period
        pop_out_len = std::max(pop_out_len, chunk_enc_len - fifo_enc_target + old_fifo_len);
        pop_out_len = std::min(pop_out_len, new_fifo_total);

        const float* pop_embs = updated_fifo.data();
        const float* pop_preds = updated_fifo_preds.data();

        update_silence_profile(pop_embs, pop_preds, pop_out_len);

        int remaining_fifo = new_fifo_total - pop_out_len;
        fifo_embs_.resize((size_t)remaining_fifo * d_model_);
        fifo_preds_.resize((size_t)remaining_fifo * n_spk_);
        if (remaining_fifo > 0) {
            std::memcpy(fifo_embs_.data(),
                        updated_fifo.data() + (size_t)pop_out_len * d_model_,
                        (size_t)remaining_fifo * d_model_ * sizeof(float));
            std::memcpy(fifo_preds_.data(),
                        updated_fifo_preds.data() + (size_t)pop_out_len * n_spk_,
                        (size_t)remaining_fifo * n_spk_ * sizeof(float));
        }
        fifo_enc_len_ = remaining_fifo;

        // Append popped frames to spkcache
        int new_sc_len = old_sc_len + pop_out_len;
        spkcache_embs_.resize((size_t)new_sc_len * d_model_);
        std::memcpy(spkcache_embs_.data() + (size_t)old_sc_len * d_model_,
                    pop_embs, (size_t)pop_out_len * d_model_ * sizeof(float));

        if (spkcache_preds_valid_) {
            spkcache_preds_.resize((size_t)new_sc_len * n_spk_);
            std::memcpy(spkcache_preds_.data() + (size_t)old_sc_len * n_spk_,
                        pop_preds, (size_t)pop_out_len * n_spk_ * sizeof(float));
        }
        spkcache_enc_len_ = new_sc_len;

        // Check if compression needed
        int target_enc = spkcache_len_ / subsampling_factor_;
        if (new_sc_len > target_enc) {
            if (!spkcache_preds_valid_) {
                // First time: init spkcache_preds from full prediction output
                spkcache_preds_.resize((size_t)new_sc_len * n_spk_);
                // Copy predictions for old spkcache frames from full_pred_out
                if (old_sc_len > 0 && full_out_len >= old_sc_len) {
                    // full_pred_out is [n_spk, full_out_len] — need to transpose to
                    // [old_sc_len, n_spk]
                    for (int t = 0; t < old_sc_len; ++t)
                        for (int s = 0; s < n_spk_; ++s)
                            spkcache_preds_[(size_t)t * n_spk_ + s] =
                                full_pred_out[(size_t)s * full_out_len + t];
                }
                // Copy pop_out predictions (already in [pop_out_len, n_spk])
                std::memcpy(spkcache_preds_.data() + (size_t)old_sc_len * n_spk_,
                            pop_preds, (size_t)pop_out_len * n_spk_ * sizeof(float));
                spkcache_preds_valid_ = true;
            }
            compress_spkcache();
        }
    } else {
        fifo_embs_ = std::move(updated_fifo);
        fifo_preds_ = std::move(updated_fifo_preds);
        fifo_enc_len_ = new_fifo_total;
    }
}

std::vector<StreamingSpeakerSegment> StreamingDiarization::postprocess_chunk(
        const std::vector<float>& probs, int n_spk, int T_out,
        float time_offset) const {

    std::vector<StreamingSpeakerSegment> segments;

    for (int s = 0; s < n_spk; ++s) {
        const float* p = probs.data() + (size_t)s * T_out;

        bool active = false;
        int start_frame = 0;

        for (int t = 0; t < T_out; ++t) {
            const bool on = (p[t] >= onset_threshold_);
            if (on && !active) {
                start_frame = t;
                active = true;
            } else if (!on && active) {
                float start_sec = time_offset + start_frame * frame_sec_;
                float end_sec   = time_offset + t * frame_sec_;
                segments.push_back({s, start_sec, end_sec});
                active = false;
            }
        }
        if (active) {
            float start_sec = time_offset + start_frame * frame_sec_;
            float end_sec   = time_offset + T_out * frame_sec_;
            segments.push_back({s, start_sec, end_sec});
        }
    }

    std::sort(segments.begin(), segments.end(),
              [](const StreamingSpeakerSegment& a, const StreamingSpeakerSegment& b) {
                  if (a.start != b.start) return a.start < b.start;
                  return a.speaker < b.speaker;
              });

    for (auto& seg : segments) {
        seg.start = std::round(seg.start * 100.0f) / 100.0f;
        seg.end   = std::round(seg.end   * 100.0f) / 100.0f;
    }

    return segments;
}

} // namespace pk
