#pragma once
// Silero VAD (https://github.com/snakers4/silero-vad, MIT licence) as a small
// standalone ggml model. One GGUF holds the 16 kHz and the 8 kHz weight sets
// (see scripts/convert_silero_vad_to_gguf.py).
//
// The model reads fixed chunks (512 samples at 16 kHz, 256 at 8 kHz). For every
// chunk it sees the last `context` samples of the previous chunk (64 or 32), so
// it needs a small state: that context plus the LSTM hidden and cell vectors.
// The state lives in SileroVad::Stream, so one loaded model can serve several
// audio streams. The loaded model is read-only; run each Stream from one thread
// at a time.
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct ggml_backend_buffer;

namespace pk {

class SileroVad {
public:
    // Returns nullptr on failure and writes a one-line reason to *err (when not null).
    static std::unique_ptr<SileroVad> load(const std::string& gguf_path, std::string* err = nullptr);
    ~SileroVad();
    SileroVad(const SileroVad&) = delete;
    SileroVad& operator=(const SileroVad&) = delete;

    // Sample rates in the file (8000 and 16000 for the official model).
    const std::vector<int>& sample_rates() const { return rates_; }
    bool supports(int sample_rate) const;
    // Samples per chunk: 512 at 16 kHz, 256 at 8 kHz. 0 for an unsupported rate.
    int chunk_samples(int sample_rate) const;
    // Seconds of audio covered by one probability.
    double chunk_sec(int sample_rate) const;

    // Streaming state for one audio stream.
    class Stream {
    public:
        // Clears the context and the LSTM state. Switching the sample rate resets too.
        void reset(int sample_rate);
        int sample_rate() const { return sr_; }

    private:
        friend class SileroVad;
        int sr_ = 0;
        std::vector<float> context_, h_, c_;
    };
    Stream new_stream(int sample_rate) const;

    // One chunk of exactly chunk_samples(sr) samples in [-1, 1]. Returns the
    // speech probability in [0, 1] and advances the state. Returns a negative
    // value on failure.
    float process_chunk(Stream& s, const float* chunk) const;

    // Whole clip, chunk by chunk, from a fresh state. The last chunk is
    // zero-padded to a full chunk, like the official audio_forward(). The result
    // has ceil(n / chunk) values.
    std::vector<float> probabilities(const float* pcm, size_t n, int sample_rate) const;

private:
    SileroVad() = default;
    struct RateWeights {
        int sr = 0, chunk = 0, context = 0, n_fft = 0, hop = 0, rpad = 0;
        ggml_tensor* stft = nullptr;
        ggml_tensor* enc_w[4] = {};
        ggml_tensor* enc_b[4] = {};
        ggml_tensor *w_ih = nullptr, *w_hh = nullptr, *b_ih = nullptr, *b_hh = nullptr;
        ggml_tensor *w_out = nullptr, *b_out = nullptr;
    };
    const RateWeights* find(int sr) const;

    std::vector<int> rates_;
    std::vector<RateWeights> w_;
    int hidden_ = 128;
    ggml_context* ctx_ = nullptr;          // tensor metadata, F32 weights
    ggml_backend_buffer* buf_ = nullptr;   // weights
};

}  // namespace pk
