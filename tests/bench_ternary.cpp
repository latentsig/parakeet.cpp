// Single-thread throughput of each ternary kernel. Usage: bench_ternary [N K T reps]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "ggml.h"
#include "ternary_kernels.hpp"

using namespace pk;

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 4096;
    const int K = argc > 2 ? std::atoi(argv[2]) : 1024;
    const int T = argc > 3 ? std::atoi(argv[3]) : 200;
    const int reps = argc > 4 ? std::atoi(argv[4]) : 20;
    std::mt19937 rng(1);
    const int nb = (K + 4) / 5, G = K / kTernaryGroup;
    std::vector<uint8_t> q((size_t)N * nb);
    for (auto& b : q) b = (uint8_t)(rng() % 243);
    std::vector<uint16_t> s((size_t)N * G, ggml_fp32_to_fp16(0.05f));
    TernaryWeight w;
    ternary_repack(q.data(), s.data(), N, K, w);
    std::vector<float> x((size_t)T * K);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& v : x) v = d(rng);
    std::vector<uint8_t> act(ternary_act_row_bytes(K) * T);
    ternary_quant_rows(x.data(), K, 0, T, act.data());
    std::vector<float> y((size_t)T * N);
    for (const TernaryKernel* k : ternary_all_kernels()) {
        k->fn(w, act.data(), T, y.data(), 0, N);  // warm up
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) k->fn(w, act.data(), T, y.data(), 0, N);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("%-8s N=%d K=%d T=%d  %8.3f ms  %7.2f GMAC/s\n", k->name, N, K, T, sec * 1e3,
                    (double)N * K * T / sec / 1e9);
    }
    return 0;
}
