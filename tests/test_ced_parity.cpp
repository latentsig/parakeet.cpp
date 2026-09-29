// In-tree check that ced.cpp built against parakeet's ggml (with its patches)
// gives the same output as standalone ced.cpp: probabilities vs ced's own
// PyTorch baseline, at the level standalone ced reaches on CPU f32 (1.7e-7).
// Needs PARAKEET_TEST_CED_GGUF and PARAKEET_TEST_CED_BASELINE. Exit 77 when
// either is unset (the baseline .baseline.gguf fixtures are gitignored in
// ced.cpp, so there is no in-tree default path).
#include "ced_tagger.hpp"
#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool load_f32(const std::string& path, const char* name, std::vector<float>& out) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{false, &ctx};
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) return false;
    ggml_tensor* t = ggml_get_tensor(ctx, name);
    bool ok = t != nullptr;
    if (ok) {
        out.resize((size_t)ggml_nelements(t));
        std::memcpy(out.data(), t->data, out.size() * sizeof(float));
    }
    gguf_free(g);
    ggml_free(ctx);
    return ok;
}

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_CED_GGUF");
    const char* base_env = std::getenv("PARAKEET_TEST_CED_BASELINE");
    if (!gguf || !base_env) {
        std::fprintf(stderr, "SKIP: PARAKEET_TEST_CED_GGUF / PARAKEET_TEST_CED_BASELINE unset\n");
        return 77;
    }
    if (!pk::CedTagger::available()) { std::fprintf(stderr, "SKIP: built without CED\n"); return 77; }
    const std::string base = base_env;
    std::vector<float> wav, ref;
    if (!load_f32(base, "audio_waveform", wav) || !load_f32(base, "probs", ref)) {
        std::fprintf(stderr, "FAIL: baseline %s\n", base.c_str());
        return 1;
    }
    if (!pk::gguf_is_ced(gguf)) { std::fprintf(stderr, "FAIL: gguf_is_ced\n"); return 1; }
    auto tagger = pk::CedTagger::load(gguf);
    if (!tagger) { std::fprintf(stderr, "FAIL: load\n"); return 1; }
    std::vector<float> probs;
    if (!tagger->scorer()(wav.data(), (int)wav.size(), probs)) { std::fprintf(stderr, "FAIL: score\n"); return 1; }
    if (probs.size() != ref.size()) { std::fprintf(stderr, "FAIL: size\n"); return 1; }
    double md = 0;
    for (size_t i = 0; i < ref.size(); ++i) md = std::fmax(md, std::fabs((double)probs[i] - ref[i]));
    std::fprintf(stderr, "max|d| = %.3e (standalone ced f32 CPU: 1.7e-7)\n", md);
    // Off-CPU the GPU matmuls move probs by ~1e-4 (ced.cpp PR #2), so only the
    // CPU gets the tight gate.
    const char* dev = std::getenv("CED_DEVICE");
    const double tol = (dev && std::strcmp(dev, "cpu") == 0) ? 1e-6 : 1e-3;
    if (md > tol) { std::fprintf(stderr, "FAIL: tol %.0e\n", tol); return 1; }
    const char* label0 = tagger->label(0);
    if (!label0 || std::strcmp(label0, "Speech") != 0) { std::fprintf(stderr, "FAIL: label 0\n"); return 1; }
    std::fprintf(stderr, "PASS\n");
    return 0;
}
