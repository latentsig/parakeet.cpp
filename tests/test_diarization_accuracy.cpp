// Diarization accuracy vs NeMo (nvidia/Nemotron-3-Diarization).
//
// Runs pk::DiarizationModel on the audio stored in a NeMo baseline
// (scripts/gen_diar_baseline.py) and checks it against NeMo's own output:
//
//   1. offline speaker probabilities: same shape, max/mean abs diff in bounds
//   2. offline segments: same count, same speakers, boundaries within 20 ms
//   3. frame-level speaker activity agreement (10 ms grid) >= 99.5%
//
// The default fixture is tests/fixtures/two_speakers.wav (LibriSpeech 1272 and
// 2086 alternating, A-B-A-B), where NeMo finds 5 segments across 2 speakers.
//
// Env:
//   PARAKEET_TEST_DIAR_GGUF       diarization GGUF (required)
//   PARAKEET_TEST_BASELINE_DIAR   baseline GGUF from gen_diar_baseline.py (required)
//   PARAKEET_TEST_DIAR_PROB_TOL   max abs prob diff (default 0.02; F32 measures
//                                 ~5e-3, quantized models need more headroom)
// Skips (77) when either required variable is unset.
#include "diarization.hpp"
#include "parity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

struct Seg { int spk; float start, end; };

std::vector<Seg> to_segs(const std::vector<float>& flat) {
    std::vector<Seg> out;
    for (size_t i = 0; i + 2 < flat.size(); i += 3)
        out.push_back({(int)flat[i], flat[i + 1], flat[i + 2]});
    std::sort(out.begin(), out.end(), [](const Seg& a, const Seg& b) {
        return a.start != b.start ? a.start < b.start : a.spk < b.spk;
    });
    return out;
}

// Speaker activity on a 10 ms grid: grid[s * T + t].
std::vector<char> to_grid(const std::vector<Seg>& segs, int n_spk, int T) {
    std::vector<char> g((size_t)n_spk * T, 0);
    for (const Seg& s : segs) {
        if (s.spk < 0 || s.spk >= n_spk) continue;
        const int a = std::max(0, (int)std::lround(s.start * 100.0f));
        const int b = std::min(T, (int)std::lround(s.end * 100.0f));
        for (int t = a; t < b; ++t) g[(size_t)s.spk * T + t] = 1;
    }
    return g;
}

}  // namespace

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    const char* base = std::getenv("PARAKEET_TEST_BASELINE_DIAR");
    if (!gguf || !base) {
        std::fprintf(stderr, "test_diarization_accuracy: PARAKEET_TEST_DIAR_GGUF and/or "
                             "PARAKEET_TEST_BASELINE_DIAR not set; skip\n");
        return 77;
    }
    const char* tol_env = std::getenv("PARAKEET_TEST_DIAR_PROB_TOL");
    const float prob_tol = tol_env ? (float)std::atof(tol_env) : 0.02f;

    auto m = pk::DiarizationModel::load(gguf);
    if (!m) { std::fprintf(stderr, "load failed: %s\n", gguf); return 1; }

    std::vector<float> audio, ref_probs, ref_segs_flat;
    std::vector<int64_t> shape;
    if (!pktest::load_baseline(base, "audio", audio, shape)) return 1;
    if (!pktest::load_baseline(base, "offline_probs", ref_probs, shape)) return 1;
    const int ref_spk = (int)shape[0], ref_T = (int)shape[1];
    if (!pktest::load_baseline(base, "offline_segs", ref_segs_flat, shape)) return 1;

    int fails = 0;

    // 1. Frame probabilities.
    std::vector<float> probs;
    int n_spk = 0, T = 0;
    m->speaker_probs(audio, probs, n_spk, T);
    std::printf("probs: ours [%d, %d], NeMo [%d, %d]\n", n_spk, T, ref_spk, ref_T);
    if (n_spk != ref_spk || T != ref_T) {
        std::fprintf(stderr, "FAIL: probability shape mismatch\n");
        return 1;
    }
    double max_diff = 0.0, sum_diff = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        const double d = std::fabs((double)probs[i] - ref_probs[i]);
        max_diff = std::max(max_diff, d);
        sum_diff += d;
    }
    const double mean_diff = sum_diff / probs.size();
    std::printf("probs: max_diff=%.5f mean_diff=%.6f (tol max %.3f, mean 0.002)\n",
                max_diff, mean_diff, prob_tol);
    if (max_diff > prob_tol || mean_diff > 2e-3) {
        std::fprintf(stderr, "FAIL: probabilities diverge from NeMo\n");
        ++fails;
    }

    // 2. Segments.
    const std::vector<Seg> ref = to_segs(ref_segs_flat);
    pk::DiarizationResult r = m->diarize_pcm(audio, 16000);
    std::vector<Seg> ours;
    for (const auto& s : r.segments) ours.push_back({s.speaker, s.start, s.end});
    std::printf("segments: ours %zu, NeMo %zu\n", ours.size(), ref.size());
    for (size_t i = 0; i < std::max(ours.size(), ref.size()); ++i) {
        const bool has_o = i < ours.size(), has_r = i < ref.size();
        std::printf("  %s spk%d %6.2f-%6.2f   NeMo spk%d %6.2f-%6.2f\n",
                    (has_o && has_r && ours[i].spk == ref[i].spk &&
                     std::fabs(ours[i].start - ref[i].start) <= 0.02f &&
                     std::fabs(ours[i].end - ref[i].end) <= 0.02f) ? "ok  " : "DIFF",
                    has_o ? ours[i].spk : -1, has_o ? ours[i].start : 0.f, has_o ? ours[i].end : 0.f,
                    has_r ? ref[i].spk : -1, has_r ? ref[i].start : 0.f, has_r ? ref[i].end : 0.f);
    }
    if (ours.size() != ref.size()) {
        std::fprintf(stderr, "FAIL: segment count differs\n");
        ++fails;
    } else {
        for (size_t i = 0; i < ref.size(); ++i) {
            if (ours[i].spk != ref[i].spk ||
                std::fabs(ours[i].start - ref[i].start) > 0.02f ||
                std::fabs(ours[i].end - ref[i].end) > 0.02f) {
                std::fprintf(stderr, "FAIL: segment %zu differs\n", i);
                ++fails;
            }
        }
    }

    // 3. Frame-level agreement over frames where either side has speech.
    const std::vector<char> go = to_grid(ours, n_spk, T), gr = to_grid(ref, n_spk, T);
    int active = 0, agree = 0;
    for (int t = 0; t < T; ++t) {
        bool any = false, same = true;
        for (int s = 0; s < n_spk; ++s) {
            const char a = go[(size_t)s * T + t], b = gr[(size_t)s * T + t];
            any = any || a || b;
            same = same && a == b;
        }
        if (any) { ++active; agree += same ? 1 : 0; }
    }
    const double agreement = active ? (double)agree / active : 1.0;
    std::printf("frame agreement: %.2f%% of %d active frames\n", 100.0 * agreement, active);
    if (agreement < 0.995) {
        std::fprintf(stderr, "FAIL: frame agreement below 99.5%%\n");
        ++fails;
    }

    std::printf(fails ? "test_diarization_accuracy: FAIL\n" : "test_diarization_accuracy: PASS\n");
    return fails ? 1 : 0;
}
