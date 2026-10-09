#pragma once
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
inline float rnnt_blank_penalty() { static const float p = env_float("PARAKEET_BLANK_PENALTY", 0.0f); return p; }
inline float ctc_blank_penalty() { static const float p = env_float("PARAKEET_CTC_BLANK_PENALTY", 0.0f); return p; }

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
