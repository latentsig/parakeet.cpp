// Streaming diarization accuracy vs NeMo cache-aware streaming
// (SortformerEncLabelModel with streaming_mode=True, the model's own chunk and
// speaker-cache config).
//
// Feeds the baseline clip through pk::StreamingDiarization two ways and checks
// both against NeMo's streaming output:
//   A. mel chunks cut from the whole-clip mel (exactly what NeMo does)
//   B. pk::StreamingMel fed with small PCM pieces (the live-audio path the
//      C-API uses), re-chunked to the model's chunk size
// Checks: per-frame probabilities (max/mean abs diff) and the segments
// (count, speaker, boundaries within 20 ms).
//
// Env: PARAKEET_TEST_DIAR_GGUF + PARAKEET_TEST_BASELINE_DIAR
//      (scripts/gen_diar_baseline.py); PARAKEET_TEST_DIAR_PROB_TOL as in
//      test_diarization_accuracy. Skips (77) when unset.
#include "diarization.hpp"
#include "diarization_streaming.hpp"
#include "mel.hpp"
#include "parity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct Run {
    std::vector<float> probs;   // [n_spk, T]
    std::vector<pk::StreamingSpeakerSegment> segs;
};

// Feed [n_mels, T] mel through the streaming diarizer in model-sized chunks.
Run stream_mel(pk::StreamingDiarization& sd, const std::vector<float>& mel, int n_mels, int T) {
    Run r;
    const int ns = sd.n_speakers(), cm = sd.chunk_mel_frames();
    r.probs.assign((size_t)ns * T, 0.0f);
    sd.reset();
    for (int lo = 0; lo < T; lo += cm) {
        const int n = std::min(cm, T - lo);
        std::vector<float> chunk((size_t)n_mels * n);
        for (int m = 0; m < n_mels; ++m)
            std::copy_n(mel.begin() + (size_t)m * T + lo, n, chunk.begin() + (size_t)m * n);
        auto segs = sd.feed_mel_chunk(chunk, n_mels, n, lo + n >= T);
        r.segs.insert(r.segs.end(), segs.begin(), segs.end());
        for (int s = 0; s < ns; ++s)
            std::copy_n(sd.last_chunk_probs().begin() + (size_t)s * n, n,
                        r.probs.begin() + (size_t)s * T + lo);
    }
    std::sort(r.segs.begin(), r.segs.end(), [](const auto& a, const auto& b) {
        return a.start != b.start ? a.start < b.start : a.speaker < b.speaker;
    });
    return r;
}

int check(const char* label, const Run& r, const std::vector<float>& ref_probs,
          const std::vector<float>& ref_segs, float tol) {
    int fails = 0;
    double max_d = 0.0, sum_d = 0.0;
    for (size_t i = 0; i < ref_probs.size(); ++i) {
        const double d = std::fabs((double)r.probs[i] - ref_probs[i]);
        max_d = std::max(max_d, d);
        sum_d += d;
    }
    const double mean_d = sum_d / ref_probs.size();
    if (std::getenv("PARAKEET_TEST_DIAR_VERBOSE")) {
        // Per-10 s max diff, to localize divergence (e.g. after cache compression).
        const size_t T = ref_probs.size() / 8;
        for (size_t lo = 0; lo < T; lo += 1000) {
            double m = 0.0;
            for (size_t sp = 0; sp < 8; ++sp)
                for (size_t t = lo; t < std::min(T, lo + 1000); ++t)
                    m = std::max(m, std::fabs((double)r.probs[sp * T + t] - ref_probs[sp * T + t]));
            std::printf("  frames %zu..%zu max_diff %.4f\n", lo, std::min(T, lo + 1000), m);
        }
    }
    std::printf("[%s] probs: max_diff=%.5f mean_diff=%.6f\n", label, max_d, mean_d);
    if (max_d > tol || mean_d > 2e-3) { std::fprintf(stderr, "[%s] FAIL: probabilities\n", label); ++fails; }

    const size_t n_ref = ref_segs.size() / 3;
    std::printf("[%s] segments: ours %zu, NeMo %zu\n", label, r.segs.size(), n_ref);
    if (r.segs.size() != n_ref) {
        std::fprintf(stderr, "[%s] FAIL: segment count\n", label);
        return fails + 1;
    }
    // NeMo's rows are sorted by (start, speaker) the same way.
    std::vector<size_t> order(n_ref);
    for (size_t i = 0; i < n_ref; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return ref_segs[a * 3 + 1] != ref_segs[b * 3 + 1] ? ref_segs[a * 3 + 1] < ref_segs[b * 3 + 1]
                                                          : ref_segs[a * 3] < ref_segs[b * 3];
    });
    for (size_t i = 0; i < n_ref; ++i) {
        const float* g = &ref_segs[order[i] * 3];
        const auto& o = r.segs[i];
        const bool ok = o.speaker == (int)g[0] && std::fabs(o.start - g[1]) <= 0.02f &&
                        std::fabs(o.end - g[2]) <= 0.02f;
        std::printf("  %s spk%d %6.2f-%6.2f   NeMo spk%d %6.2f-%6.2f\n", ok ? "ok  " : "DIFF",
                    o.speaker, o.start, o.end, (int)g[0], g[1], g[2]);
        if (!ok) ++fails;
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

    std::vector<float> audio, ref_probs, ref_segs;
    std::vector<int64_t> shape;
    if (!pktest::load_baseline(base, "audio", audio, shape)) return 1;
    if (!pktest::load_baseline(base, "stream_probs", ref_probs, shape)) return 1;
    const int T = (int)shape[1];
    if (!pktest::load_baseline(base, "stream_segs", ref_segs, shape)) return 1;

    pk::StreamingDiarization sd(m->loader());
    const int n_mels = sd.n_mels();
    int fails = 0;

    // A. Whole-clip mel (no peak normalization in streaming mode), trimmed to
    //    floor(S / hop) frames like NeMo.
    {
        std::vector<float> mel;
        int nm = 0, Tm = 0;
        m->mel().compute(audio, mel, nm, Tm);
        if (Tm < T) { std::fprintf(stderr, "mel too short: %d < %d\n", Tm, T); return 1; }
        std::vector<float> trimmed((size_t)nm * T);
        for (int i = 0; i < nm; ++i)
            std::copy_n(mel.begin() + (size_t)i * Tm, T, trimmed.begin() + (size_t)i * T);
        fails += check("whole-clip mel", stream_mel(sd, trimmed, nm, T), ref_probs, ref_segs, tol);
    }

    // B. Incremental mel from 100 ms PCM pieces.
    {
        pk::StreamingMel sm(m->loader());
        std::vector<std::vector<float>> cols;   // per-frame mel columns
        auto take = [&](const std::vector<float>& fm, int n) {
            for (int t = 0; t < n; ++t) {
                std::vector<float> col(n_mels);
                for (int i = 0; i < n_mels; ++i) col[i] = fm[(size_t)i * n + t];
                cols.push_back(std::move(col));
            }
        };
        for (size_t lo = 0; lo < audio.size(); lo += 1600) {
            const int n = (int)std::min<size_t>(1600, audio.size() - lo);
            int nf = 0;
            auto fm = sm.feed(audio.data() + lo, n, nf);
            take(fm, nf);
        }
        int nf = 0;
        auto tail = sm.finalize(nf);
        take(tail, nf);
        if ((int)cols.size() < T) { std::fprintf(stderr, "stream mel too short\n"); return 1; }
        std::vector<float> mel((size_t)n_mels * T);
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < n_mels; ++i) mel[(size_t)i * T + t] = cols[t][i];
        fails += check("StreamingMel", stream_mel(sd, mel, n_mels, T), ref_probs, ref_segs, tol);
    }

    std::printf(fails ? "test_streaming_diarization: FAIL\n" : "test_streaming_diarization: PASS\n");
    return fails ? 1 : 0;
}
