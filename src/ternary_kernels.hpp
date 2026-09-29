#pragma once
#include <vector>

#include "ternary.hpp"

namespace pk {

using TernaryRowsFn = void (*)(const TernaryWeight&, const uint8_t* act, int T, float* y,
                               int r0, int r1);
struct TernaryKernel {
    const char* name;
    TernaryRowsFn fn;
};

using TernaryQuantFn = void (*)(const float* x, int K, int t0, int t1, uint8_t* act);
struct TernaryQuant {
    const char* name;
    TernaryQuantFn fn;
};

const TernaryKernel* ternary_kernel_x86_vnni();
const TernaryKernel* ternary_kernel_x86_avx2();
const TernaryKernel* ternary_kernel_neon();
std::vector<const TernaryKernel*> ternary_all_kernels();

const TernaryQuant* ternary_quant_x86_avx512();
const TernaryQuant* ternary_quant_x86_avx2();
std::vector<const TernaryQuant*> ternary_all_quants();

}  // namespace pk
