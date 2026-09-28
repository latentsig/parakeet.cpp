#pragma once
#include "model_loader.hpp"
#include "diarization_encoder.hpp"
#include "diarization_head.hpp"
#include <vector>

namespace pk {

// A finished speaker segment on the stream's timeline (seconds from stream start).
struct StreamingSpeakerSegment {
    int speaker;
    float start;
    float end;
};

// StreamingDiarization — NeMo Sortformer cache-aware streaming ("AOSC"),
// synchronous mode (SortformerEncLabelModel.forward_streaming_step +
// SortformerModules.streaming_update), for nvidia/Nemotron-3-Diarization.
//
// Each chunk of mel frames is pre-encoded (FeatureStacking + projection +
// embed_norm) and the transformer + speaker head run over
// [speaker cache | FIFO | chunk]. The chunk's slice of the high-resolution
// output is the result for that chunk; the downsampled predictions drive the
// FIFO -> speaker-cache update and the score-based cache compression that
// keeps the speaker identities stable across chunks.
//
// chunk_len / spkcache_len / fifo_len / spkcache_update_period come from the
// GGUF and are in ENCODER frames (80 ms), as in NeMo. With the Nemotron-3
// config a chunk is 264 encoder frames = 2112 mel frames = 21.12 s.
//
// The mel must be the un-normalized log-mel of the stream (NeMo does not
// peak-normalize in streaming mode), e.g. from pk::StreamingMel.
class StreamingDiarization {
public:
    explicit StreamingDiarization(const ModelLoader& ml);

    void reset();

    // Feed the next chunk: row-major [n_mels, n_frames] (mel[m*n_frames + t]),
    // 0 < n_frames <= chunk_mel_frames(). Only the final chunk may be short.
    // Returns the speaker segments that ENDED in this chunk; with is_last,
    // every still-open segment is closed at the end of the stream.
    std::vector<StreamingSpeakerSegment> feed_mel_chunk(
        const std::vector<float>& mel, int n_mels, int n_frames, bool is_last);

    // Speaker probabilities of the last fed chunk, speaker-major
    // [n_speakers, last_chunk_frames()] (one frame per mel frame, 10 ms).
    const std::vector<float>& last_chunk_probs() const { return last_probs_; }
    int last_chunk_frames() const { return last_frames_; }

    // Segments that are still active at the current end of the stream, with
    // `end` set to the stream time consumed so far.
    std::vector<StreamingSpeakerSegment> open_segments() const;

    int chunk_mel_frames() const { return chunk_len_ * subsampling_; }
    int n_mels() const { return n_mels_; }
    int n_speakers() const { return n_spk_; }
    float frame_sec() const { return frame_sec_; }
    // Mel frames consumed so far.
    long long frames_done() const { return frames_done_; }

private:
    void streaming_update(const std::vector<float>& chunk_emb, int chunk_frames,
                          const std::vector<float>& preds, int spkcache_frames,
                          int fifo_frames);
    void compress_spkcache();
    void track_segments(const std::vector<float>& probs, int n_frames, bool is_last,
                        std::vector<StreamingSpeakerSegment>& out);

    const ModelLoader& ml_;
    DiarizationEncoder encoder_;
    DiarizationHead head_;

    int d_model_, n_spk_, subsampling_, upsample_, n_mels_;
    int chunk_len_, spkcache_len_, fifo_len_, update_period_, sil_frames_per_spk_;
    float frame_sec_, onset_, offset_;
    float sil_threshold_, pred_score_threshold_, scores_boost_latest_;
    float strong_boost_rate_, weak_boost_rate_, min_pos_scores_rate_;
    bool use_learnable_sil_emb_ = false;
    std::vector<float> learnable_sil_emb_;   // [d_model]

    // Streaming state. Embeddings are time-major [frames, d_model];
    // predictions are [frames, n_spk] at encoder resolution.
    std::vector<float> spkcache_, spkcache_preds_;
    std::vector<float> fifo_, fifo_preds_;
    bool spkcache_compressed_ = false;
    std::vector<float> mean_sil_emb_;        // [d_model]
    long long n_sil_frames_ = 0;

    // Output state.
    long long frames_done_ = 0;              // mel frames consumed
    std::vector<float> last_probs_;
    int last_frames_ = 0;
    std::vector<char> active_;               // per speaker
    std::vector<long long> start_frame_;     // per speaker, when active
};

} // namespace pk
