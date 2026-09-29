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

const TernaryKernel* ternary_kernel_x86_vnni();
const TernaryKernel* ternary_kernel_x86_avx2();
const TernaryKernel* ternary_kernel_neon();
std::vector<const TernaryKernel*> ternary_all_kernels();

}  // namespace pk
