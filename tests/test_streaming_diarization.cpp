// Streaming diarization accuracy vs NeMo cache-aware streaming
// (SortformerEncLabelModel with streaming_mode=True).
//
// For the checkpoint's own configuration and every model-card latency mode
// present in the baseline (stream_probs_<mode>; scripts/gen_diar_baseline.py),
// feeds the clip through pk::StreamingDiarization two ways and checks both
// against NeMo:
//   A. the whole-clip mel in one feed (NeMo computes the mel up front)
//   B. pk::StreamingMel fed with 100 ms PCM pieces, each piece's frames fed as
//      they arrive (the live path the C-API uses)
// Checks: per-frame probabilities (max/mean abs diff) and the segments
// (count, speaker, boundaries within 20 ms).
//
// Env: PARAKEET_TEST_DIAR_GGUF + PARAKEET_TEST_BASELINE_DIAR; optional
//      PARAKEET_TEST_DIAR_PROB_TOL (default 0.02) and PARAKEET_TEST_DIAR_VERBOSE
//      (per-10 s max diff). Skips (77) when the required ones are unset.
#include "diarization.hpp"
#include "diarization_streaming.hpp"
#include "mel.hpp"
#include "parity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Run {
    std::vector<float> probs;   // [n_spk, T]
    std::vector<pk::StreamingSpeakerSegment> segs;
};

// Collect one feed_mel call's output into the run.
void collect(pk::StreamingDiarization& sd, Run& r, long long& done, int T,
             const std::vector<pk::StreamingSpeakerSegment>& segs) {
    const int ns = sd.n_speakers(), n = sd.last_frames();
    for (int s = 0; s < ns; ++s)
        for (int t = 0; t < n && done + t < T; ++t)
            r.probs[(size_t)s * T + done + t] = sd.last_probs()[(size_t)s * n + t];
    done += n;
    r.segs.insert(r.segs.end(), segs.begin(), segs.end());
}

void sort_segs(Run& r) {
    std::sort(r.segs.begin(), r.segs.end(), [](const auto& a, const auto& b) {
        return a.start != b.start ? a.start < b.start : a.speaker < b.speaker;
    });
}

// A: whole-clip mel [n_mels, T] in one feed.
Run whole_clip(pk::StreamingDiarization& sd, const std::vector<float>& mel, int n_mels, int T) {
    Run r;
    r.probs.assign((size_t)sd.n_speakers() * T, 0.0f);
    long long done = 0;
    sd.reset();
    collect(sd, r, done, T, sd.feed_mel(mel, n_mels, T, true));
    sort_segs(r);
    return r;
}

// B: live, StreamingMel over 100 ms PCM pieces; frames beyond T are dropped
// like NeMo's floor(S / hop) length.
Run live(pk::StreamingDiarization& sd, const pk::ModelLoader& ml,
         const std::vector<float>& audio, int T) {
    Run r;
    r.probs.assign((size_t)sd.n_speakers() * T, 0.0f);
    long long done = 0, fed = 0;
    sd.reset();
    pk::StreamingMel sm(ml);
    const int n_mels = sd.n_mels();
    auto feed = [&](std::vector<float> fm, int nf, bool last) {
        const int keep = (int)std::max(0LL, std::min<long long>(nf, T - fed));
        if (keep < nf) {
            std::vector<float> cut((size_t)n_mels * keep);
            for (int m = 0; m < n_mels; ++m)
                std::copy_n(fm.begin() + (size_t)m * nf, keep, cut.begin() + (size_t)m * keep);
            fm.swap(cut);
        }
        fed += keep;
        collect(sd, r, done, T, sd.feed_mel(fm, n_mels, keep, last));
    };
    for (size_t lo = 0; lo < audio.size(); lo += 1600) {
        const int n = (int)std::min<size_t>(1600, audio.size() - lo);
        int nf = 0;
        std::vector<float> fm = sm.feed(audio.data() + lo, n, nf);
        feed(std::move(fm), nf, false);
    }
    int nf = 0;
    std::vector<float> tail = sm.finalize(nf);
    feed(std::move(tail), nf, true);
    sort_segs(r);
    return r;
}

int check(const std::string& label, const Run& r, const std::vector<float>& ref_probs,
          const std::vector<float>& ref_segs, float tol) {
    int fails = 0;
    double max_d = 0.0, sum_d = 0.0;
    for (size_t i = 0; i < ref_probs.size(); ++i) {
        const double d = std::fabs((double)r.probs[i] - ref_probs[i]);
        max_d = std::max(max_d, d);
        sum_d += d;
    }
    const double mean_d = sum_d / ref_probs.size();
    std::printf("[%s] probs: max_diff=%.5f mean_diff=%.6f\n", label.c_str(), max_d, mean_d);
    if (std::getenv("PARAKEET_TEST_DIAR_VERBOSE")) {
        const size_t T = ref_probs.size() / 8;
        for (size_t lo = 0; lo < T; lo += 1000) {
            double m = 0.0;
            for (size_t sp = 0; sp < 8; ++sp)
                for (size_t t = lo; t < std::min(T, lo + 1000); ++t)
                    m = std::max(m, std::fabs((double)r.probs[sp * T + t] - ref_probs[sp * T + t]));
            std::printf("  frames %zu..%zu max_diff %.4f\n", lo, std::min(T, lo + 1000), m);
        }
    }
    if (max_d > tol || mean_d > 2e-3) {
        std::fprintf(stderr, "[%s] FAIL: probabilities\n", label.c_str());
        ++fails;
    }

    const size_t n_ref = ref_segs.size() / 3;
    std::printf("[%s] segments: ours %zu, NeMo %zu\n", label.c_str(), r.segs.size(), n_ref);
    std::vector<size_t> order(n_ref);
    for (size_t i = 0; i < n_ref; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return ref_segs[a * 3 + 1] != ref_segs[b * 3 + 1] ? ref_segs[a * 3 + 1] < ref_segs[b * 3 + 1]
                                                          : ref_segs[a * 3] < ref_segs[b * 3];
    });
    for (size_t i = 0; i < std::max(n_ref, r.segs.size()); ++i) {
        const bool hr = i < n_ref, ho = i < r.segs.size();
        const float* g = hr ? &ref_segs[order[i] * 3] : nullptr;
        const bool ok = hr && ho && r.segs[i].speaker == (int)g[0] &&
                        std::fabs(r.segs[i].start - g[1]) <= 0.02f &&
                        std::fabs(r.segs[i].end - g[2]) <= 0.02f;
        if (!ok) {
            std::printf("  DIFF spk%d %6.2f-%6.2f   NeMo spk%d %6.2f-%6.2f\n",
                        ho ? r.segs[i].speaker : -1, ho ? r.segs[i].start : 0.f,
                        ho ? r.segs[i].end : 0.f, hr ? (int)g[0] : -1,
                        hr ? g[1] : 0.f, hr ? g[2] : 0.f);
            ++fails;
        }
    }
    return fails;
}

}  // namespace

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_DIAR");
    if (!gguf || !base) {
        std::fprintf(stderr, "test_streaming_diarization: PARAKEET_TEST_DIAR_GGUF and/or "
                             "PARAKEET_TEST_BASELINE_DIAR not set; skip\n");
        return 77;
    }
    const char* tol_env = std::getenv("PARAKEET_TEST_DIAR_PROB_TOL");
    const float tol = tol_env ? (float)std::atof(tol_env) : 0.02f;

    auto m = pk::DiarizationModel::load(gguf);
    if (!m) { std::fprintf(stderr, "load failed: %s\n", gguf); return 1; }
    const pk::ModelLoader& ml = m->loader();

    std::vector<float> audio;
    std::vector<int64_t> shape;
    if (!pktest::load_baseline(base, "audio", audio, shape)) return 1;

    // Whole-clip mel (no peak normalization in streaming mode), trimmed to
    // floor(S / hop) frames like NeMo.
    std::vector<float> mel_full;
    int n_mels = 0, Tm = 0;
    m->mel().compute(audio, mel_full, n_mels, Tm);
    const int T = std::min(Tm, (int)(audio.size() / ml.config().hop_length));
    std::vector<float> mel((size_t)n_mels * T);
    for (int i = 0; i < n_mels; ++i)
        std::copy_n(mel_full.begin() + (size_t)i * Tm, T, mel.begin() + (size_t)i * T);

    const struct { const char* key; pk::DiarLatency latency; } modes[] = {
        {"", pk::DiarLatency::Model},
        {"_low", pk::DiarLatency::Low},
        {"_very_low", pk::DiarLatency::VeryLow},
        {"_ultra_low", pk::DiarLatency::UltraLow},
    };
    int fails = 0, checked = 0;
    for (const auto& mode : modes) {
        std::vector<float> ref_probs, ref_segs;
        std::vector<int64_t> ps;
        const std::string pk_ = std::string("stream_probs") + mode.key;
        const std::string sk_ = std::string("stream_segs") + mode.key;
        if (!pktest::load_baseline(base, pk_, ref_probs, ps)) {
            std::printf("[%s] not in the baseline; skipped\n", pk_.c_str());
            continue;
        }
        if (!pktest::load_baseline(base, sk_, ref_segs, shape)) return 1;
        if ((int)ps[1] != T) {
            std::fprintf(stderr, "[%s] length %lld != %d\n", pk_.c_str(), (long long)ps[1], T);
            return 1;
        }
        pk::StreamingDiarization sd(ml, pk::diar_stream_config(mode.latency, ml.config()));
        const std::string name = mode.key[0] ? std::string(mode.key + 1) : "model";
        fails += check(name + " whole-clip", whole_clip(sd, mel, n_mels, T), ref_probs, ref_segs, tol);
        fails += check(name + " live", live(sd, ml, audio, T), ref_probs, ref_segs, tol);
        ++checked;
    }
    if (checked == 0) { std::fprintf(stderr, "no streaming baseline found\n"); return 1; }
    std::printf(fails ? "test_streaming_diarization: FAIL\n" : "test_streaming_diarization: PASS\n");
    return fails ? 1 : 0;
}
