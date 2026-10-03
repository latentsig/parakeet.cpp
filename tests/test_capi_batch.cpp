#include "model.hpp"
#include "audio_io.hpp"
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

// Batch-API B=1 equivalence smoke test.
//
// Loads the real-speech fixture, transcribes it single-clip, then runs it as a
// 2-clip batch of the same audio. Asserts both batch results equal the single
// result, proving the batch path gives the same transcript as the single path
// (text equality; batched floats are close, not bit-identical). This is
// self-consistency (our own code on both sides), so it only needs
// PARAKEET_TEST_GGUF.
int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) { std::fprintf(stderr, "env not set; skip\n"); return 77; }
    auto model = pk::Model::load(gguf);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }

    pk::Audio audio;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", audio) || audio.samples.empty()) {
        std::fprintf(stderr, "wav load failed\n");
        return 1;
    }
    const std::vector<float>& pcm = audio.samples;

    std::string single = model->transcribe_pcm(pcm, 16000);
    auto batch = model->transcribe_pcm_batch({pcm, pcm}, 16000);
    if (batch.size() != 2) { std::fprintf(stderr, "batch size %zu\n", batch.size()); return 1; }
    if (batch[0] != single || batch[1] != single) {
        std::fprintf(stderr, "MISMATCH single='%s' b0='%s' b1='%s'\n",
                     single.c_str(), batch[0].c_str(), batch[1].c_str());
        return 1;
    }
    std::fprintf(stderr, "OK batch==single: '%s'\n", single.c_str());
    return 0;
}
