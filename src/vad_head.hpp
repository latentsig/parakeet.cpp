#pragma once
// Voice-activity head that moondream/parakeet-ultra and -redux carry on the
// encoder's subsampler output: proj (1x1 conv) -> ctx (conv, k=5) -> out (1x1
// conv) -> sigmoid, one probability per 80 ms frame.
#include <vector>

namespace pk {

class ModelLoader;

struct VadWeights {
    int d_in = 0, hidden = 0, kernel = 0;
    std::vector<float> proj_w, proj_b;   // [hidden][d_in], [hidden]
    std::vector<float> ctx_w, ctx_b;     // [hidden][hidden][kernel], [hidden]
    std::vector<float> out_w;            // [hidden]
    float out_b = 0.0f;
};

// The three choices the checkpoint does not document. Fixed from evidence in
// docs/ternary.md (section "VAD head wiring").
struct VadVariant {
    bool relu_after_proj = true;
    bool residual = false;
    bool relu_after_ctx = true;
    static VadVariant from_index(int i) {
        VadVariant v;
        v.relu_after_proj = (i & 1) != 0;
        v.residual = (i & 2) != 0;
        v.relu_after_ctx = (i & 4) != 0;
        return v;
    }
};

class VadHead {
public:
    // Throws std::runtime_error if the GGUF has no VAD head or its tensors are
    // missing or mis-shaped.
    explicit VadHead(const ModelLoader& ml);
    // x: row-major [T][d_in]. Returns T probabilities in [0, 1].
    std::vector<float> probabilities(const float* x, int T, const VadVariant* v = nullptr) const;
    float frame_sec() const { return frame_sec_; }
    static std::vector<float> run(const VadWeights& w, const VadVariant& v, const float* x, int T);

private:
    VadWeights w_;
    float frame_sec_ = 0.08f;
};

}  // namespace pk
