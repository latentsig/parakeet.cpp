// test_diarization_parity.cpp — dump intermediate outputs for parity comparison.
//
// Env: PARAKEET_TEST_DIAR_GGUF (required), PARAKEET_TEST_DIAR_WAV (required),
//      PARAKEET_TEST_DIAR_OUT (required, output directory).
#include "diarization.hpp"
#include "mel.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "audio_io.hpp"
#include "diarization_encoder.hpp"
#include "diarization_head.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

static void write_npy_f32(const std::string& path, const float* data,
                          const std::vector<int64_t>& shape) {
    // NumPy v1 format: magic(6) + version(2) + header_len(2) + header
    // Header = dict_string + padding_spaces + \n, total padded to multiple of 64.
    std::string magic = "\x93NUMPY";
    uint8_t version[2] = {1, 0};
    std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) dict += ", ";
        dict += std::to_string(shape[i]);
    }
    if (shape.size() == 1) dict += ",";
    dict += "), }";
    // header = dict + padding + \n, total must be multiple of 64
    // total_file = 6 (magic) + 2 (version) + 2 (hlen) + header_len
    // We want header_len such that 10 + header_len is multiple of 64.
    // header_len = dict.size() + n_pad + 1(\n)
    int overhead = 10;  // magic + version + hlen
    int target = overhead + dict.size() + 1;  // +1 for \n
    int padded = ((target + 63) / 64) * 64;
    int n_pad = padded - target;
    uint16_t hlen = (uint16_t)(dict.size() + n_pad + 1);

    std::ofstream f(path, std::ios::binary);
    f.write(magic.data(), 6);
    f.write((char*)version, 2);
    f.write((char*)&hlen, 2);
    f.write(dict.data(), (std::streamsize)dict.size());
    for (int i = 0; i < n_pad; ++i) f.write(" ", 1);
    f.write("\n", 1);
    size_t n = 1;
    for (auto s : shape) n *= s;
    f.write((const char*)data, (std::streamsize)(n * sizeof(float)));
}

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_DIAR_GGUF");
    const char* wav_path = std::getenv("PARAKEET_TEST_DIAR_WAV");
    const char* out_dir = std::getenv("PARAKEET_TEST_DIAR_OUT");
    if (!gguf || !wav_path || !out_dir) {
        std::fprintf(stderr, "PARAKEET_TEST_DIAR_GGUF, PARAKEET_TEST_DIAR_WAV, "
                      "PARAKEET_TEST_DIAR_OUT all required\n");
        return 77;
    }

    // Load model
    std::unique_ptr<pk::DiarizationModel> m = pk::DiarizationModel::load(gguf);
    if (!m) {
        std::fprintf(stderr, "DiarizationModel::load failed\n");
        return 1;
    }

    // Load audio
    pk::Audio audio;
    if (!pk::load_audio_16k_mono(wav_path, audio)) {
        std::fprintf(stderr, "failed to load audio: %s\n", wav_path);
        return 1;
    }
    std::printf("audio: %zu samples, %.2fs\n", audio.samples.size(),
                (float)audio.samples.size() / 16000.0f);

    // Build components from the loader (same as DiarizationModel::run)
    const pk::ModelLoader& ml = m->loader();
    pk::MelFrontend mel(ml);
    pk::DiarizationEncoder encoder(ml);
    pk::DiarizationHead head(ml);

    // 1. Mel
    std::vector<float> feats;
    int n_mels = 0, T = 0;
    mel.compute(audio.samples, feats, n_mels, T);
    std::printf("mel: [%d, %d]\n", n_mels, T);
    write_npy_f32(std::string(out_dir) + "/mel_pk.npy", feats.data(),
                  {(int64_t)n_mels, (int64_t)T});

    // 2. Encoder
    std::vector<float> enc_out;
    int d_model = 0, T_enc = 0;
    encoder.forward(feats, n_mels, T, enc_out, d_model, T_enc);
    std::printf("enc_out: [%d, %d]\n", d_model, T_enc);
    write_npy_f32(std::string(out_dir) + "/enc_out_pk.npy", enc_out.data(),
                  {(int64_t)d_model, (int64_t)T_enc});

    // 3. Head
    std::vector<float> probs;
    int n_spk = 0, T_out = 0;
    head.forward(enc_out, d_model, T_enc, probs, n_spk, T_out);
    std::printf("probs: [%d, %d]\n", n_spk, T_out);
    write_npy_f32(std::string(out_dir) + "/probs_pk.npy", probs.data(),
                  {(int64_t)n_spk, (int64_t)T_out});

    // 4. Segments (reuse the model's diarize_path which does the full pipeline)
    pk::DiarizationResult r = m->diarize_path(wav_path);
    std::printf("segments: %zu\n", r.segments.size());

    std::ofstream sf(std::string(out_dir) + "/segments_pk.json");
    sf << "[\n";
    for (size_t i = 0; i < r.segments.size(); ++i) {
        sf << "  {\"speaker\": " << r.segments[i].speaker
           << ", \"start\": " << r.segments[i].start
           << ", \"end\": " << r.segments[i].end << "}";
        if (i + 1 < r.segments.size()) sf << ",";
        sf << "\n";
    }
    sf << "]\n";

    for (size_t i = 0; i < r.segments.size() && i < 20; ++i) {
        std::printf("  spk %d: %.2f - %.2f\n",
                    r.segments[i].speaker,
                    r.segments[i].start, r.segments[i].end);
    }

    std::printf("\nDone. Outputs in %s/\n", out_dir);
    return 0;
}
