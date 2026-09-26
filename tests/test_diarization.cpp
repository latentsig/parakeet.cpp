#include "diarization.hpp"
#include "model_loader.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

int main() {
    const char* path = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    if (!path) {
        std::fprintf(stderr,
            "PARAKEET_TEST_DIAR_GGUF not set; skipping diarization test\n");
        return 77;
    }

    // Load via DiarizationModel::load (exercises loader, mel, encoder, head).
    std::unique_ptr<pk::DiarizationModel> m = pk::DiarizationModel::load(path);
    if (!m) {
        std::fprintf(stderr, "DiarizationModel::load failed for %s\n", path);
        return 1;
    }

    // Config sanity: arch must be "diarization" and diarization.present true.
    pk::ModelLoader ml;
    if (!ml.load(path)) {
        std::fprintf(stderr, "ModelLoader::load failed\n");
        return 1;
    }
    const pk::ParakeetConfig& c = ml.config();
    if (c.arch != "diarization") {
        std::fprintf(stderr, "arch != diarization (got %s)\n", c.arch.c_str());
        return 1;
    }
    if (!c.diarization.present) {
        std::fprintf(stderr, "diarization.present is false\n");
        return 1;
    }
    if (c.diarization.n_speakers == 0) {
        std::fprintf(stderr, "n_speakers == 0\n");
        return 1;
    }
    std::printf("diarization config OK: arch=%s n_spk=%u tf_d_model=%u "
                "upsample=%u frame_sec=%.4f onset=%.2f offset=%.2f\n",
                c.arch.c_str(), c.diarization.n_speakers,
                c.diarization.tf_d_model, c.diarization.upsample_factor,
                c.diarization.frame_resolution_sec,
                c.diarization.onset_threshold,
                c.diarization.offset_threshold);

    // Verify key sortformer tensors are present.
    const char* required[] = {
        "sortformer_modules.encoder_proj.weight",
        "sortformer_modules.subpixel_upsample.weight",
        "sortformer_modules.first_hidden_to_hidden.weight",
        "sortformer_modules.single_hidden_to_spks.weight",
        nullptr,
    };
    for (size_t i = 0; required[i]; ++i) {
        if (ml.tensor(required[i]) == nullptr) {
            std::fprintf(stderr, "missing tensor: %s\n", required[i]);
            return 1;
        }
    }
    std::printf("all required sortformer tensors present\n");

    // If a test audio file is provided, run end-to-end diarization.
    const char* wav = std::getenv("PARAKEET_TEST_DIAR_WAV");
    if (wav) {
        try {
            pk::DiarizationResult r = m->diarize_path(wav);
            std::printf("diarized %s -> %zu segments, %d speakers\n",
                        wav, r.segments.size(), r.n_speakers);
            for (size_t i = 0; i < r.segments.size() && i < 20; ++i) {
                std::printf("  spk %d: %.2f - %.2f\n",
                            r.segments[i].speaker,
                            r.segments[i].start, r.segments[i].end);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "diarize_path threw: %s\n", e.what());
            return 1;
        }
    } else {
        std::printf("PARAKEET_TEST_DIAR_WAV not set; skipping inference test\n");
    }

    return 0;
}
