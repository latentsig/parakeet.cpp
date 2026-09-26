#pragma once
#include "model_loader.hpp"
#include "diarization_encoder.hpp"
#include "diarization_head.hpp"
#include "mel.hpp"
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace pk {

// A speaker segment emitted during streaming. Timestamps are wall-clock seconds
// from the start of the stream.
struct StreamingSpeakerSegment {
    int speaker;
    float start;
    float end;
};

// Streaming diarization state for the Nemotron-3-Diarization Sortformer.
//
// The AOSC ("Attention-Only Streaming with Compression") mechanism:
//
//  1. Audio is processed in chunks of `chunk_len` mel frames (264).
//  2. Each chunk is concatenated with the speaker cache: [spkcache | chunk].
//     The full bidirectional encoder runs over this concatenated sequence.
//  3. The encoder output for the chunk portion (last chunk_enc_len frames)
//     is passed through the diarization head to get per-frame speaker probs.
//  4. The probs for the spkcache portion are stored for compression scoring.
//  5. After processing, the chunk embeddings are appended to a FIFO buffer.
//     When the FIFO overflows (exceeds fifo_len), the oldest frames are popped
//     and appended to the spkcache.
//  6. When the spkcache exceeds spkcache_len, AOSC compression is triggered:
//     frames are scored per-speaker (log-odds), top-K are selected globally,
//     and the spkcache is rebuilt to spkcache_len frames.
//
// For Nemotron-3-Diarization:
//   fifo_len=0, spkcache_len=264, chunk_len=264
//   subsampling_factor=8, upsample_factor=8
//   use_learnable_sil_emb=true
//
// With fifo_len=0, every chunk immediately overflows the FIFO, so every chunk's
// embeddings are appended to the spkcache and compression runs after every chunk.
//
// The streaming path reuses the SAME encoder and head as the offline path —
// only the chunking and spkcache management differ.
class StreamingDiarization {
public:
    explicit StreamingDiarization(const ModelLoader& ml);
    ~StreamingDiarization();

    // Reset the stream state (clear spkcache, frame counter).
    void reset();

    // Feed one chunk of mel features [n_mels, n_frames] (row-major:
    // mel[m*n_frames + t]). The caller must provide exactly `chunk_len` frames
    // (or fewer for the final chunk). Returns the speaker segments for this
    // chunk (with wall-clock timestamps).
    //
    // is_last marks the final chunk — the spkcache is not updated after it.
    std::vector<StreamingSpeakerSegment> feed_mel_chunk(
        const std::vector<float>& mel_chunk, int n_mels, int n_frames,
        bool is_last = false);

    // Chunk parameters (from GGUF config).
    int chunk_len() const { return chunk_len_; }
    int spkcache_len() const { return spkcache_len_; }
    int fifo_len() const { return fifo_len_; }
    int n_mels() const { return n_mels_; }
    int n_speakers() const { return n_spk_; }

    // The frame-to-second conversion: each output frame is hop_length/sample_rate
    // seconds = 160/16000 = 0.01s.
    float frame_sec() const { return frame_sec_; }

private:
    const ModelLoader& ml_;
    DiarizationEncoder encoder_;
    DiarizationHead head_;

    int d_model_;            // encoder d_model (512)
    int tf_d_model_;         // sortformer tf_d_model (192)
    int n_spk_;              // number of speakers (8)
    int n_layers_;           // transformer blocks (31)
    int n_heads_;            // attention heads (8)
    int head_dim_;           // d_model / n_heads (64)
    int ff_dim_;             // feed-forward dim (2048)
    int subsampling_factor_; // 8
    int upsample_factor_;    // 8
    int n_mels_;             // 128
    int chunk_len_;          // 264 mel frames
    int spkcache_len_;       // 264 mel frames
    int fifo_len_;           // 0 for Nemotron-3
    float frame_sec_;        // 0.01s per output frame
    float onset_threshold_;
    float offset_threshold_;

    // --- AOSC streaming config ---
    int spkcache_sil_frames_per_spk_; // 3
    float sil_threshold_;              // 0.2
    float pred_score_threshold_;       // 0.25
    float scores_boost_latest_;       // 0.05
    float strong_boost_rate_;         // 0.75
    float weak_boost_rate_;           // 1.5
    float min_pos_scores_rate_;       // 0.5
    int max_index_;                   // 99999 (placeholder for disabled slots)

    // --- Spkcache state ---
    // Encoder embeddings [d_model, spkcache_enc_len] (channels-first: emb[c*len + t])
    std::vector<float> spkcache_embs_;
    std::vector<float> spkcache_preds_;  // [n_spk, spkcache_enc_len]
    int spkcache_enc_len_ = 0;
    bool spkcache_preds_valid_ = false;

    // --- FIFO state ---
    // With fifo_len=0, this overflows every chunk.
    std::vector<float> fifo_embs_;    // [d_model, fifo_enc_len]
    std::vector<float> fifo_preds_;   // [n_spk, fifo_enc_len]
    int fifo_enc_len_ = 0;

    // --- Silence profile ---
    std::vector<float> mean_sil_emb_;  // [d_model]
    int n_sil_frames_ = 0;

    // --- The learned silence embedding (model parameter) ---
    bool has_silence_emb_ = false;
    std::vector<float> silence_emb_;  // [d_model]

    // Running count of total mel frames consumed (for wall-clock timestamps).
    int total_mel_frames_ = 0;

    // --- Internal helpers ---

    // AOSC: boost top-K scores per speaker
    void boost_topk_scores(float* scores, int n_frames, int n_spk,
                           int k_per_spk, float scale_factor, float offset) const;

    // AOSC: compress spkcache from current length to spkcache_len_ frames.
    void compress_spkcache();

    // Update running silence profile from popped embeddings.
    void update_silence_profile(const float* pop_embs, const float* pop_preds,
                                 int pop_len);

    // Update stream state after processing one chunk (FIFO → spkcache → compress).
    // chunk_preenc: pre-encoded embeddings for the chunk [d_model, chunk_enc_len]
    // chunk_preds: per-speaker probs for the chunk [n_spk, chunk_out_len]
    // full_pred_out: full prediction output for [spkcache | chunk] [n_spk, total_out_len]
    //                (used to extract spkcache predictions for scoring)
    void stream_state_update(
        const std::vector<float>& chunk_preenc, int chunk_enc_len,
        const std::vector<float>& chunk_preds, int chunk_out_len,
        const std::vector<float>& full_pred_out, int full_out_len);

    // Post-process per-frame speaker probabilities into segments for this chunk.
    std::vector<StreamingSpeakerSegment> postprocess_chunk(
        const std::vector<float>& probs, int n_spk, int T_out,
        float time_offset) const;
};

} // namespace pk
