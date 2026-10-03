// Timing for batched decode. Not a ctest. Usage:
//   bench_batch_decode decode <model.gguf> <clip.wav> [N=16]
//     decode only: N encoder outputs (cut at different lengths from the clip), decoded one
//     by one with tdt_greedy / rnnt_greedy and in one transducer_greedy_batch call.
// Prints "<label> <milliseconds>" lines: the best of 3 runs after one warm-up run.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "audio_io.hpp"
#include "encoder.hpp"
#include "joint.hpp"
#include "mel.hpp"
#include "model.hpp"
#include "model_loader.hpp"
#include "prediction.hpp"
#include "rnnt.hpp"
#include "tdt.hpp"
#include "transducer_batch.hpp"

using Clock = std::chrono::steady_clock;

template <class F> static double best_ms(F&& f) {
    f();  // warm-up
    double best = 1e30;
    for (int i = 0; i < 3; ++i) {
        const auto t0 = Clock::now();
        f();
        best = std::min(best, std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    return best;
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s decode model wav [N]\n", argv[0]); return 2; }
    if (std::string(argv[1]) != "decode") { std::fprintf(stderr, "unknown mode %s\n", argv[1]); return 2; }
    pk::Audio a;
    if (!pk::load_audio_16k_mono(argv[3], a)) { std::fprintf(stderr, "wav load failed\n"); return 1; }
    const int N = argc > 4 ? std::atoi(argv[4]) : 16;
    pk::ModelLoader ml;
    if (!ml.load(argv[2])) return 1;
    const auto& cfg = ml.config();
    pk::Encoder enc(ml);
    pk::MelFrontend melfe(ml);
    std::vector<std::vector<float>> encs(N);
    std::vector<int> Ts(N);
    int dm = 0;
    for (int n = 0; n < N; ++n) {
        const size_t len = a.samples.size() * (size_t)(60 + (n * 7) % 41) / 100;  // 60..100 percent
        std::vector<float> pcm(a.samples.begin(), a.samples.begin() + len), mel, eo;
        int n_mels = 0, T0 = 0, to = 0;
        melfe.compute(pcm, mel, n_mels, T0);
        enc.forward(mel, n_mels, T0, eo, dm, to);
        encs[n].resize((size_t)to * dm);
        for (int t = 0; t < to; ++t)
            for (int c = 0; c < dm; ++c) encs[n][(size_t)t * dm + c] = eo[(size_t)c * to + t];
        Ts[n] = to;
    }
    pk::PredictionNet pred(ml);
    pk::Joint joint(ml);
    const bool tdt = !cfg.tdt_durations.empty();
    const int blank = (int)cfg.blank_id, maxs = (int)cfg.max_symbols;
    size_t tok_item = 0, tok_batch = 0;
    std::vector<std::vector<int32_t>> ref(N);
    const double item_ms = best_ms([&] {
        tok_item = 0;
        for (int n = 0; n < N; ++n) {
            ref[n] = tdt ? pk::tdt_greedy(pred, joint, encs[n], Ts[n], dm, cfg.tdt_durations, blank, maxs)
                         : pk::rnnt_greedy(pred, joint, encs[n], Ts[n], dm, blank, maxs);
            tok_item += ref[n].size();
        }
    });
    std::vector<std::vector<int32_t>> ids;
    const double batch_ms = best_ms([&] {
        pk::transducer_greedy_batch(pred, joint, encs, Ts, dm, cfg.tdt_durations, blank, maxs, ids, nullptr);
        tok_batch = 0;
        for (auto& v : ids) tok_batch += v.size();
    });
    int same = 0;
    for (int n = 0; n < N; ++n) same += ids[n] == ref[n];
    std::printf("decode_per_item %.1f\ndecode_batch %.1f\n(N %d, tokens per item path %zu, batch %zu, "
                "%d of %d items with identical token ids)\n",
                item_ms, batch_ms, N, tok_item, tok_batch, same, N);
    return 0;
}
