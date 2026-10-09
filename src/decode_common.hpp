#pragma once
#include <atomic>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <cmath>
namespace pk {
// argmax over a[0..n): first index of the max (matches torch.max tie-break).
// RT Captions patch: decode-time knobs, read once from the environment.
//   PARAKEET_BLANK_PENALTY      subtracted from the RNN-T blank logit before argmax (default 0)
//   PARAKEET_CTC_BLANK_PENALTY  subtracted from the CTC blank log-prob before argmax (default 0)
//   PARAKEET_STREAM_DECODER     "ctc" = streaming sessions decode with the CTC head of a hybrid
//                               *_ctc model instead of RNN-T (default: RNN-T)
// A penalty > 0 counters a model that emits blank too often (dropped words).
inline float env_float(const char* name, float fallback) {
    const char* v = std::getenv(name);
    if (!v || !*v) return fallback;
    char* end = nullptr;
    const float f = std::strtof(v, &end);
    return end != v ? f : fallback;
}
// Read on every call (a getenv is ~microseconds; decoding runs a few hundred steps a second),
// so an app can change them between sessions without reloading the library.
struct DecodeOptions {
    std::atomic<bool> set{false};      // true once a host called parakeet_capi_set_decode_options
    std::atomic<bool> stream_ctc{false};
    std::atomic<float> rnnt_blank_penalty{0.0f};
    std::atomic<float> ctc_blank_penalty{0.0f};
    std::atomic<int> lookahead_frames{-2};   // -2 = never set (use env), -1 = model default
    // Token ids the greedy decoders may never pick (e.g. a script lock: no Devanagari in
    // English mode). Set between sessions from the host's thread; read while decoding.
    std::vector<int32_t> suppressed;
};

inline DecodeOptions& decode_options() { static DecodeOptions o; return o; }
// Forbid the suppressed ids in one decision's scores (logits or log-probs).
inline void apply_suppression(float* scores, int n) {
    for (int32_t id : decode_options().suppressed)
        if (id >= 0 && id < n) scores[id] = -INFINITY;
}
inline float rnnt_blank_penalty() {
    const DecodeOptions& o = decode_options();
    return o.set ? o.rnnt_blank_penalty.load() : env_float("PARAKEET_BLANK_PENALTY", 0.0f);
}
inline float ctc_blank_penalty() {
    const DecodeOptions& o = decode_options();
    return o.set ? o.ctc_blank_penalty.load() : env_float("PARAKEET_CTC_BLANK_PENALTY", 0.0f);
}
// Streaming look-ahead override in encoder frames (att_context_right); -1 = the model's own.
inline int stream_lookahead_frames() {
    const int v = decode_options().lookahead_frames.load();
    if (v != -2) return v;
    const char* e = std::getenv("PARAKEET_LOOKAHEAD_FRAMES");
    return (e && *e) ? std::atoi(e) : -1;
}

inline bool stream_decoder_is_ctc() {
    const DecodeOptions& o = decode_options();
    if (o.set) return o.stream_ctc.load();
    const char* v = std::getenv("PARAKEET_STREAM_DECODER");
    return v && v[0] == 'c' && v[1] == 't' && v[2] == 'c' && v[3] == 0;
}

inline int decode_argmax(const float* a, int n) {
    int best = 0; float bv = a[0];
    for (int i = 1; i < n; ++i) if (a[i] > bv) { bv = a[i]; best = i; }
    return best;
}
// NeMo rescaled max_prob confidence over a[0..n) at index k:
//   conf = (N*p_max - 1)/(N - 1), p_max = softmax(a)[k]. Stable softmax.
inline float decode_max_prob_conf(const float* a, int n, int k) {
    float mx = a[0];
    for (int i = 1; i < n; ++i) if (a[i] > mx) mx = a[i];
    double denom = 0.0;
    for (int i = 0; i < n; ++i) denom += std::exp((double)a[i] - (double)mx);
    const double p_max = std::exp((double)a[k] - (double)mx) / denom;
    const double N = (double)n;
    return (float)((N * p_max - 1.0) / (N - 1.0));
}
} // namespace pk
