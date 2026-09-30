#pragma once
#include "prediction.hpp"
#include "joint.hpp"
#include "decode_types.hpp"
#include <vector>
#include <cstdint>
namespace pk {
// Batched greedy decode for N utterances. encs[n]: row-major [T[n], enc_hidden].
// durations empty -> RNNT (advance-by-1); non-empty -> TDT (advance-by-duration).
// Outputs per item: ids[n], and (if toks != nullptr) TokenInfo[n]. Applies the
// same per-item decision rule as rnnt_greedy / tdt_greedy, but it is NOT
// guaranteed bit-identical to them: for N > 1 ggml picks a different matmul
// kernel than for N = 1, so prediction and joint outputs differ in float
// (logits by up to about 1e-4 measured on the packed Redux model). The emitted
// token sequences were identical on the test clips; a near-tie argmax could in
// principle flip. The tests compare with a tolerance.
void transducer_greedy_batch(
    const PredictionNet& pred, const Joint& joint,
    const std::vector<std::vector<float>>& encs,
    const std::vector<int>& T,             // [N] per-item frame counts
    int enc_hidden,
    const std::vector<int32_t>& durations, // empty=RNNT
    int blank_id, int max_symbols,
    std::vector<std::vector<int32_t>>& ids,            // OUT [N][.]
    std::vector<std::vector<TokenInfo>>* toks);         // OUT [N][.] or nullptr
} // namespace pk
