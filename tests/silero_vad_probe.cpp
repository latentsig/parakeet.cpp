// Prints the Silero VAD speech probability of every chunk of an audio file.
//
//   silero_vad_probe <silero.gguf> <audio.wav | audio.f32> [--sr 16000|8000] [--reps N] [--quiet]
//
// A .wav is loaded as 16 kHz mono (and resampled to 8 kHz with --sr 8000). A
// .f32 file is raw little-endian float32 mono PCM at the --sr rate, used to give
// the C++ and the ONNX reference the very same samples. Not a ctest: with
// --reps it also times the run (single thread) for a speed number.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "audio_io.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "silero_vad.hpp"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.gguf audio.{wav,f32} [--sr N] [--reps N] [--quiet]\n", argv[0]);
        return 2;
    }
    int sr = 16000, reps = 1;
    bool quiet = false;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--sr") && i + 1 < argc) sr = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--quiet")) quiet = true;
    }
    pk::set_num_threads(1);
    std::string err;
    auto vad = pk::SileroVad::load(argv[1], &err);
    if (!vad) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    std::vector<float> pcm;
    const std::string path = argv[2];
    if (path.size() > 4 && path.substr(path.size() - 4) == ".f32") {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return 1;
        std::fseek(f, 0, SEEK_END);
        pcm.resize((size_t)std::ftell(f) / sizeof(float));
        std::fseek(f, 0, SEEK_SET);
        if (std::fread(pcm.data(), sizeof(float), pcm.size(), f) != pcm.size()) return 1;
        std::fclose(f);
    } else {
        pk::Audio a;
        if (!pk::load_audio_16k_mono(path, a)) {
            std::fprintf(stderr, "cannot read %s\n", path.c_str());
            return 1;
        }
        pcm = sr == 16000 ? a.samples : pk::resample_linear(a.samples, 16000, sr);
    }
    std::vector<float> p = vad->probabilities(pcm.data(), pcm.size(), sr);  // also warms up
    if (p.empty()) {
        std::fprintf(stderr, "inference failed\n");
        return 1;
    }
    if (!quiet)
        for (float v : p) std::printf("%.9g\n", (double)v);
    if (reps > 1) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) vad->probabilities(pcm.data(), pcm.size(), sr);
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
        const double audio = (double)pcm.size() / sr;
        std::fprintf(stderr, "chunks=%zu audio=%.2fs time=%.4fs rtf=%.5f (%.0fx realtime) per_chunk=%.1fus\n", p.size(),
                     audio, sec, sec / audio, audio / sec, sec / p.size() * 1e6);
    }
    pk::shutdown_backend();
    return 0;
}
