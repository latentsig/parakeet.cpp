// Silero VAD ggml port: structure, streaming == batch, and parity with an ONNX
// reference when one is given.
//
//   PARAKEET_TEST_SILERO_GGUF   F32 silero-vad GGUF (else the test skips, exit 77)
//   PARAKEET_TEST_SILERO_PCM    raw float32 mono 16 kHz audio, a multiple of 512 samples (optional)
//   PARAKEET_TEST_SILERO_REF    text file with the onnxruntime probability of each chunk of that audio (optional)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "audio_io.hpp"
#include "ggml_graph.hpp"
#include "silero_vad.hpp"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_SILERO_GGUF");
    if (!gguf) return 77;
    pk::set_num_threads(1);
    std::string err;
    auto vad = pk::SileroVad::load(gguf, &err);
    if (!vad) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    CHECK(vad->supports(16000) && vad->supports(8000) && !vad->supports(44100));
    CHECK(vad->chunk_samples(16000) == 512 && vad->chunk_samples(8000) == 256 && vad->chunk_samples(22050) == 0);

    // Streaming one chunk at a time gives the same values as the whole-clip call.
    pk::Audio a;
    CHECK(pk::load_audio_16k_mono("tests/fixtures/speech.wav", a));
    const std::vector<float> all = vad->probabilities(a.samples.data(), a.samples.size(), 16000);
    CHECK(all.size() == (a.samples.size() + 511) / 512);
    pk::SileroVad::Stream s = vad->new_stream(16000);
    std::vector<float> chunk(512);
    for (size_t i = 0; i < all.size(); ++i) {
        std::fill(chunk.begin(), chunk.end(), 0.0f);
        for (size_t j = 0; j < 512 && i * 512 + j < a.samples.size(); ++j) chunk[j] = a.samples[i * 512 + j];
        CHECK(vad->process_chunk(s, chunk.data()) == all[i]);
    }
    // Speech clip: mostly speech. Digital silence: no speech.
    int speech = 0;
    for (float p : all) {
        CHECK(p >= 0.0f && p <= 1.0f);
        speech += p >= 0.5f;
    }
    CHECK(speech * 2 > (int)all.size());
    std::vector<float> zeros(16000, 0.0f);
    for (float p : vad->probabilities(zeros.data(), zeros.size(), 16000)) CHECK(p < 0.5f);
    // reset() restores the initial state.
    s.reset(16000);
    CHECK(vad->process_chunk(s, chunk.data()) >= 0.0f);
    s.reset(16000);
    std::fill(chunk.begin(), chunk.end(), 0.0f);
    for (size_t j = 0; j < 512; ++j) chunk[j] = a.samples[j];
    CHECK(vad->process_chunk(s, chunk.data()) == all[0]);

    const char* pcm_path = std::getenv("PARAKEET_TEST_SILERO_PCM");
    const char* ref_path = std::getenv("PARAKEET_TEST_SILERO_REF");
    if (pcm_path && ref_path) {
        std::ifstream f(pcm_path, std::ios::binary | std::ios::ate);
        std::vector<float> pcm((size_t)f.tellg() / sizeof(float));
        f.seekg(0);
        f.read((char*)pcm.data(), (std::streamsize)(pcm.size() * sizeof(float)));
        std::vector<float> ref;
        std::ifstream r(ref_path);
        for (double v; r >> v;) ref.push_back((float)v);
        const std::vector<float> got = vad->probabilities(pcm.data(), pcm.size(), 16000);
        CHECK(got.size() == ref.size());
        double mx = 0.0;
        for (size_t i = 0; i < got.size() && i < ref.size(); ++i) mx = std::fmax(mx, std::fabs(got[i] - ref[i]));
        std::printf("max abs diff vs onnxruntime: %.3g over %zu chunks\n", mx, got.size());
        CHECK(mx < 1e-5);
    }
    pk::shutdown_backend();
    if (failures) return 1;
    std::printf("test_silero_vad OK\n");
    return 0;
}
