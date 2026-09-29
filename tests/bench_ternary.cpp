// Single-thread throughput of each ternary kernel, next to ggml's own mul_mat
// with Q8_0 and F16 weights of the same shape (1 thread, activations F32, so the
// ggml rows include their per-call activation quantization; the ternary rows do
// not include ternary_quant_rows, which is timed on its own). Usage: bench_ternary [N K T reps]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "ggml.h"
#include "ggml-cpu.h"
#include "ternary_kernels.hpp"

using namespace pk;

// Time ggml_mul_mat(W[K,N] of type wt, X[K,T] f32) on one thread; returns seconds per run.
static double time_ggml_mul_mat(ggml_type wt, int N, int K, int T, int reps, const std::vector<float>& x,
                                std::mt19937& rng) {
    ggml_init_params ip{};
    ip.mem_size = (size_t)N * K * 4 + (size_t)T * K * 4 + (size_t)T * N * 4 + (64u << 20);
    ip.no_alloc = false;
    ggml_context* ctx = ggml_init(ip);
    ggml_tensor* w = ggml_new_tensor_2d(ctx, wt, K, N);
    std::vector<float> wf((size_t)N * K);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& v : wf) v = d(rng);
    if (wt == GGML_TYPE_F32) std::memcpy(w->data, wf.data(), ggml_nbytes(w));
    else if (wt == GGML_TYPE_F16) ggml_fp32_to_fp16_row(wf.data(), (ggml_fp16_t*)w->data, (int64_t)N * K);
    else ggml_quantize_chunk(wt, wf.data(), w->data, 0, N, K, nullptr);
    ggml_tensor* xt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, T);
    std::memcpy(xt->data, x.data(), ggml_nbytes(xt));
    ggml_tensor* y = ggml_mul_mat(ctx, w, xt);
    ggml_cgraph* g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, y);
    ggml_cplan plan = ggml_graph_plan(g, 1, nullptr);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.data();
    ggml_graph_compute(g, &plan);  // warm up
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) ggml_graph_compute(g, &plan);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    ggml_free(ctx);
    return sec;
}

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
    {
        // activation quantization (op 1 of ternary_linear), per call
        ternary_quant_rows(x.data(), K, 0, T, act.data());
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) ternary_quant_rows(x.data(), K, 0, T, act.data());
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::printf("quant    N=-    K=%d T=%d  %8.3f ms  (ternary_quant_rows, not in the kernel rows)\n", K, T,
                    sec * 1e3);
    }
    for (const TernaryKernel* k : ternary_all_kernels()) {
        // the scalar reference is 50x slower; a few runs are enough for it
        const int kr = std::strcmp(k->name, "scalar") ? reps : std::max(1, reps / 10);
        k->fn(w, act.data(), T, y.data(), 0, N);  // warm up
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < kr; ++r) k->fn(w, act.data(), T, y.data(), 0, N);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / kr;
        std::printf("%-8s N=%d K=%d T=%d  %8.3f ms  %7.2f GMAC/s\n", k->name, N, K, T, sec * 1e3,
                    (double)N * K * T / sec / 1e9);
    }
    for (ggml_type wt : {GGML_TYPE_Q8_0, GGML_TYPE_F16}) {
        const double sec = time_ggml_mul_mat(wt, N, K, T, reps, x, rng);
        std::printf("ggml_%-3s N=%d K=%d T=%d  %8.3f ms  %7.2f GMAC/s\n", ggml_type_name(wt), N, K, T, sec * 1e3,
                    (double)N * K * T / sec / 1e9);
    }
    return 0;
}
