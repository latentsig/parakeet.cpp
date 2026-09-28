#include "diarization_streaming.hpp"
#include "backend.hpp"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace pk {

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();

// Append rows [lo, hi) of a row-major [*, width] buffer to `dst`.
void append_rows(std::vector<float>& dst, const std::vector<float>& src, int width,
                 int lo, int hi) {
    dst.insert(dst.end(), src.begin() + (size_t)lo * width, src.begin() + (size_t)hi * width);
}

// Drop the first `n` rows of a row-major [*, width] buffer.
void drop_rows(std::vector<float>& v, int width, int n) {
    v.erase(v.begin(), v.begin() + (size_t)n * width);
}

float round2(float x) { return std::round(x * 100.0f) / 100.0f; }

}  // namespace

StreamingDiarization::StreamingDiarization(const ModelLoader& ml)
    : ml_(ml), encoder_(ml), head_(ml) {
    const auto& cfg = ml.config();
    const auto& d = cfg.diarization;
    d_model_       = (int)cfg.d_model;
    n_spk_         = (int)d.n_speakers;
    subsampling_   = encoder_.subsampling();
    upsample_      = (int)d.upsample_factor;
    n_mels_        = encoder_.n_mels();
    chunk_len_     = d.chunk_len;
    spkcache_len_  = d.spkcache_len;
    fifo_len_      = d.fifo_len;
    update_period_ = d.spkcache_update_period;
    sil_frames_per_spk_ = d.spkcache_sil_frames_per_spk;
    frame_sec_     = d.frame_resolution_sec;
    onset_         = d.onset_threshold;
    offset_        = d.offset_threshold;
    sil_threshold_        = d.sil_threshold;
    pred_score_threshold_ = d.pred_score_threshold;
    scores_boost_latest_  = d.scores_boost_latest;
    strong_boost_rate_    = d.strong_boost_rate;
    weak_boost_rate_      = d.weak_boost_rate;
    min_pos_scores_rate_  = d.min_pos_scores_rate;

    if (chunk_len_ <= 0 || spkcache_len_ <= 0 || fifo_len_ < 0 || update_period_ <= 0 ||
        upsample_ != subsampling_ || spkcache_len_ / n_spk_ - sil_frames_per_spk_ <= 0) {
        throw std::runtime_error("parakeet: invalid diarization streaming config");
    }

    if (d.use_learnable_sil_emb) {
        const ggml_tensor* t = ml.tensor("sortformer_modules.learnable_sil_emb");
        if (!t || t->type != GGML_TYPE_F32 || ggml_nelements(t) != d_model_)
            throw std::runtime_error("parakeet: learnable_sil_emb missing or not F32");
        ensure_weights_realized(ml);
        learnable_sil_emb_.resize(d_model_);
        ggml_backend_tensor_get(t, learnable_sil_emb_.data(), 0, d_model_ * sizeof(float));
        use_learnable_sil_emb_ = true;
    }
    reset();
}

void StreamingDiarization::reset() {
    spkcache_.clear(); spkcache_preds_.clear();
    fifo_.clear(); fifo_preds_.clear();
    spkcache_compressed_ = false;
    mean_sil_emb_.assign(d_model_, 0.0f);
    n_sil_frames_ = 0;
    frames_done_ = 0;
    last_probs_.clear();
    last_frames_ = 0;
    active_.assign(n_spk_, 0);
    start_frame_.assign(n_spk_, 0);
}

std::vector<StreamingSpeakerSegment> StreamingDiarization::feed_mel_chunk(
        const std::vector<float>& mel, int n_mels, int n_frames, bool is_last) {
    if (n_mels != n_mels_ || n_frames <= 0 || n_frames > chunk_mel_frames() ||
        mel.size() != (size_t)n_mels * n_frames) {
        throw std::runtime_error("parakeet: bad streaming diarization chunk shape");
    }

    // 1. Pre-encode the chunk -> [cl, d].
    std::vector<float> chunk_emb;
    int cl = 0;
    encoder_.pre_encode(mel, n_mels, n_frames, chunk_emb, cl);

    // 2. Transformer + head over [spkcache | fifo | chunk].
    const int S = (int)(spkcache_.size() / d_model_);
    const int F = (int)(fifo_.size() / d_model_);
    const int total = S + F + cl;
    std::vector<float> seq;
    seq.reserve((size_t)total * d_model_);
    seq.insert(seq.end(), spkcache_.begin(), spkcache_.end());
    seq.insert(seq.end(), fifo_.begin(), fifo_.end());
    seq.insert(seq.end(), chunk_emb.begin(), chunk_emb.end());

    std::vector<float> enc;
    encoder_.transformer_forward(seq, total, enc);
    std::vector<float> hp;   // [n_spk, total * up]
    int n_spk = 0, T_hr = 0;
    head_.forward(enc, total, hp, n_spk, T_hr);

    // 3. Encoder-resolution predictions [total, n_spk]: mean over each block of
    //    `up` high-resolution frames (NeMo downsample_preds).
    std::vector<float> preds((size_t)total * n_spk_);
    for (int t = 0; t < total; ++t)
        for (int s = 0; s < n_spk_; ++s) {
            double acc = 0.0;
            for (int u = 0; u < upsample_; ++u) acc += hp[(size_t)s * T_hr + t * upsample_ + u];
            preds[(size_t)t * n_spk_ + s] = (float)(acc / upsample_);
        }

    // 4. This chunk's high-resolution slice, trimmed to the real mel frames.
    const int base = (S + F) * upsample_;
    last_frames_ = n_frames;
    last_probs_.resize((size_t)n_spk_ * n_frames);
    for (int s = 0; s < n_spk_; ++s)
        std::copy_n(hp.begin() + (size_t)s * T_hr + base, n_frames,
                    last_probs_.begin() + (size_t)s * n_frames);

    // 5. Cache update for the next chunk.
    if (!is_last) streaming_update(chunk_emb, cl, preds, S, F);

    std::vector<StreamingSpeakerSegment> out;
    track_segments(last_probs_, n_frames, is_last, out);
    return out;
}

// SortformerModules.streaming_update (sync mode, lc = rc = 0).
void StreamingDiarization::streaming_update(const std::vector<float>& chunk_emb,
                                            int cl, const std::vector<float>& preds,
                                            int S, int F) {
    const int d = d_model_, ns = n_spk_;
    // FIFO predictions are refreshed from this step's output.
    fifo_preds_.assign(preds.begin() + (size_t)S * ns, preds.begin() + (size_t)(S + F) * ns);
    fifo_.insert(fifo_.end(), chunk_emb.begin(), chunk_emb.end());
    append_rows(fifo_preds_, preds, ns, S + F, S + F + cl);

    if (F + cl <= fifo_len_) return;

    int pop = std::max(update_period_, cl - fifo_len_ + F);
    pop = std::min(pop, F + cl);

    if (!use_learnable_sil_emb_) {
        // _get_silence_profile: running mean of popped frames whose summed
        // speaker probability is below sil_threshold.
        std::vector<double> sum(d, 0.0);
        long long count = 0;
        for (int t = 0; t < pop; ++t) {
            float p = 0.0f;
            for (int s = 0; s < ns; ++s) p += fifo_preds_[(size_t)t * ns + s];
            if (p < sil_threshold_) {
                ++count;
                for (int c = 0; c < d; ++c) sum[c] += fifo_[(size_t)t * d + c];
            }
        }
        if (count > 0) {
            const long long n_new = n_sil_frames_ + count;
            for (int c = 0; c < d; ++c)
                mean_sil_emb_[c] = (float)(((double)mean_sil_emb_[c] * n_sil_frames_ + sum[c]) / n_new);
            n_sil_frames_ = n_new;
        }
    }

    if (!spkcache_compressed_) {
        // Until the first compression the cache predictions are this step's.
        spkcache_preds_.assign(preds.begin(), preds.begin() + (size_t)S * ns);
    }
    append_rows(spkcache_, fifo_, d, 0, pop);
    append_rows(spkcache_preds_, fifo_preds_, ns, 0, pop);
    drop_rows(fifo_, d, pop);
    drop_rows(fifo_preds_, ns, pop);

    if ((int)(spkcache_.size() / d) > spkcache_len_) {
        compress_spkcache();
        spkcache_compressed_ = true;
    }
}

// SortformerModules._compress_spkcache (eval: no speaker permutation, no noise).
void StreamingDiarization::compress_spkcache() {
    const int d = d_model_, ns = n_spk_;
    const int n = (int)(spkcache_preds_.size() / ns);
    const int per_spk = spkcache_len_ / ns - sil_frames_per_spk_;
    const int strong_k = (int)std::floor(per_spk * strong_boost_rate_);
    const int weak_k   = (int)std::floor(per_spk * weak_boost_rate_);
    const int min_pos  = (int)std::floor(per_spk * min_pos_scores_rate_);
    const float* P = spkcache_preds_.data();

    // _get_log_pred_scores
    std::vector<float> sc((size_t)n * ns);
    for (int t = 0; t < n; ++t) {
        float log1_sum = 0.0f;
        for (int s = 0; s < ns; ++s)
            log1_sum += std::log(std::max(1.0f - P[(size_t)t * ns + s], pred_score_threshold_));
        for (int s = 0; s < ns; ++s) {
            const float p = P[(size_t)t * ns + s];
            sc[(size_t)t * ns + s] = std::log(std::max(p, pred_score_threshold_)) -
                                     std::log(std::max(1.0f - p, pred_score_threshold_)) +
                                     log1_sum - std::log(0.5f);
        }
    }

    // _disable_low_scores
    for (int s = 0; s < ns; ++s) {
        int pos = 0;
        for (int t = 0; t < n; ++t) {
            float& v = sc[(size_t)t * ns + s];
            if (!(P[(size_t)t * ns + s] > 0.5f)) v = -kInf;
            if (v > 0.0f) ++pos;
        }
        if (pos >= min_pos)
            for (int t = 0; t < n; ++t) {
                float& v = sc[(size_t)t * ns + s];
                if (!(v > 0.0f) && P[(size_t)t * ns + s] > 0.5f) v = -kInf;
            }
    }

    // Boost frames newly added since the last compression.
    if (scores_boost_latest_ > 0.0f)
        for (size_t i = (size_t)spkcache_len_ * ns; i < sc.size(); ++i) sc[i] += scores_boost_latest_;

    // _boost_topk_scores: add -scale*log(0.5) to each speaker's top-k frames.
    auto boost = [&](int k, float scale) {
        k = std::min(k, n);
        if (k <= 0) return;
        const float add = -scale * std::log(0.5f);
        std::vector<int> idx(n);
        for (int s = 0; s < ns; ++s) {
            for (int t = 0; t < n; ++t) idx[t] = t;
            // Ties (common: confident frames clamp to the same score) go to
            // the earlier frame, so the result is deterministic.
            std::nth_element(idx.begin(), idx.begin() + (k - 1), idx.end(), [&](int a, int b) {
                const float va = sc[(size_t)a * ns + s], vb = sc[(size_t)b * ns + s];
                return va != vb ? va > vb : a < b;
            });
            for (int i = 0; i < k; ++i) sc[(size_t)idx[i] * ns + s] += add;
        }
    };
    boost(strong_k, 2.0f);
    boost(weak_k, 1.0f);

    // Append sil_frames_per_spk frames of +inf per speaker (reserved silence slots).
    const int n_tot = n + sil_frames_per_spk_;
    // _get_topk_indices over the speaker-major flattening (index = s*n_tot + t).
    std::vector<std::pair<float, int>> flat((size_t)ns * n_tot);
    for (int s = 0; s < ns; ++s)
        for (int t = 0; t < n_tot; ++t)
            flat[(size_t)s * n_tot + t] = {t < n ? sc[(size_t)t * ns + s] : kInf, s * n_tot + t};
    const int K = spkcache_len_;
    std::nth_element(flat.begin(), flat.begin() + (K - 1), flat.end(),
                     [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                         return a.first != b.first ? a.first > b.first : a.second < b.second;
                     });
    constexpr int kMaxIndex = std::numeric_limits<int>::max();
    std::vector<int> top(K);
    for (int i = 0; i < K; ++i) top[i] = flat[i].first == -kInf ? kMaxIndex : flat[i].second;
    std::sort(top.begin(), top.end());

    // _gather_spkcache_and_preds
    const std::vector<float>& sil = use_learnable_sil_emb_ ? learnable_sil_emb_ : mean_sil_emb_;
    std::vector<float> new_embs((size_t)K * d), new_preds((size_t)K * ns, 0.0f);
    for (int i = 0; i < K; ++i) {
        const int t = top[i] == kMaxIndex ? -1 : top[i] % n_tot;
        if (t < 0 || t >= n) {
            std::copy(sil.begin(), sil.end(), new_embs.begin() + (size_t)i * d);
        } else {
            std::copy_n(spkcache_.begin() + (size_t)t * d, d, new_embs.begin() + (size_t)i * d);
            std::copy_n(spkcache_preds_.begin() + (size_t)t * ns, ns, new_preds.begin() + (size_t)i * ns);
        }
    }
    spkcache_.swap(new_embs);
    spkcache_preds_.swap(new_preds);
}

// Hysteresis binarization carried across chunks, matching the offline
// DiarizationModel::postprocess on the concatenated probabilities.
void StreamingDiarization::track_segments(const std::vector<float>& probs, int n_frames,
                                          bool is_last,
                                          std::vector<StreamingSpeakerSegment>& out) {
    const size_t first = out.size();
    for (int s = 0; s < n_spk_; ++s) {
        const float* p = probs.data() + (size_t)s * n_frames;
        for (int t = 0; t < n_frames; ++t) {
            const long long f = frames_done_ + t;
            if (!active_[s] && p[t] >= onset_) {
                active_[s] = 1;
                start_frame_[s] = f;
            } else if (active_[s] && p[t] < offset_) {
                active_[s] = 0;
                out.push_back({s, round2(start_frame_[s] * frame_sec_), round2(f * frame_sec_)});
            }
        }
    }
    frames_done_ += n_frames;
    if (is_last) {
        for (int s = 0; s < n_spk_; ++s)
            if (active_[s]) {
                active_[s] = 0;
                out.push_back({s, round2(start_frame_[s] * frame_sec_),
                               round2(frames_done_ * frame_sec_)});
            }
    }
    std::sort(out.begin() + first, out.end(),
              [](const StreamingSpeakerSegment& a, const StreamingSpeakerSegment& b) {
                  return a.start != b.start ? a.start < b.start : a.speaker < b.speaker;
              });
}

std::vector<StreamingSpeakerSegment> StreamingDiarization::open_segments() const {
    std::vector<StreamingSpeakerSegment> out;
    for (int s = 0; s < n_spk_; ++s)
        if (active_[s])
            out.push_back({s, round2(start_frame_[s] * frame_sec_), round2(frames_done_ * frame_sec_)});
    return out;
}

} // namespace pk
