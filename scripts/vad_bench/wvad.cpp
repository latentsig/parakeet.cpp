// Harness over whisper.cpp's public whisper_vad_* API. Not part of whisper.cpp.
//   wvad probs <model> <threads> <wav>...   -> one JSON line per wav (probs + segments)
//   wvad time  <model> <threads> <reps> <wav> -> detect_speech seconds per rep
#include "whisper.h"
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
static bool read_wav16(const char* p, std::vector<float>& out) {
    FILE* f = fopen(p, "rb"); if (!f) return false;
    char id[4]; uint32_t sz; fread(id, 1, 4, f); fread(&sz, 4, 1, f); fread(id, 1, 4, f);
    while (fread(id, 1, 4, f) == 4 && fread(&sz, 4, 1, f) == 1) {
        if (!memcmp(id, "data", 4)) {
            std::vector<int16_t> d(sz / 2); fread(d.data(), 2, d.size(), f); fclose(f);
            out.resize(d.size()); for (size_t i = 0; i < d.size(); i++) out[i] = d[i] / 32768.0f; return true;
        }
        fseek(f, sz, SEEK_CUR);
    }
    fclose(f); return false;
}
static void segs(whisper_vad_context* v, whisper_vad_params p, const char* key) {
    whisper_vad_segments* s = whisper_vad_segments_from_probs(v, p);
    printf("\"%s\":[", key);
    int n = s ? whisper_vad_segments_n_segments(s) : 0;
    for (int i = 0; i < n; i++)
        printf("%s[%.2f,%.2f]", i ? "," : "", whisper_vad_segments_get_segment_t0(s, i) / 100.0, whisper_vad_segments_get_segment_t1(s, i) / 100.0);
    printf("]"); if (s) whisper_vad_free_segments(s);
}
static void quiet(enum ggml_log_level, const char*, void*) {}
int main(int argc, char** argv) {
    whisper_log_set(quiet, nullptr);
    if (argc < 5) return 1;
    std::string mode = argv[1];
    whisper_vad_context_params cp = whisper_vad_default_context_params();
    cp.n_threads = atoi(argv[3]); cp.use_gpu = false;
    whisper_vad_context* v = whisper_vad_init_from_file_with_params(argv[2], cp);
    if (!v) { fprintf(stderr, "load failed\n"); return 2; }
    std::vector<float> y;
    if (mode == "time") {
        int reps = atoi(argv[4]);
        if (!read_wav16(argv[5], y)) return 3;
        for (int r = 0; r < reps; r++) {
            auto t0 = std::chrono::steady_clock::now();
            whisper_vad_detect_speech(v, y.data(), (int)y.size());
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("%.4f\n", dt);
        }
        return 0;
    }
    for (int a = 4; a < argc; a++) {
        if (!read_wav16(argv[a], y)) return 3;
        if (!whisper_vad_detect_speech(v, y.data(), (int)y.size())) return 4;
        int n = whisper_vad_n_probs(v); float* pr = whisper_vad_probs(v);
        printf("{\"name\":\"%s\",\"n\":%d,\"probs\":[", argv[a], n);
        for (int i = 0; i < n; i++) printf("%s%.6f", i ? "," : "", pr[i]);
        printf("],");
        segs(v, whisper_vad_default_params(), "default"); printf(",");
        whisper_vad_params m = whisper_vad_default_params();
        m.threshold = 0.5f; m.min_speech_duration_ms = 100; m.min_silence_duration_ms = 200; m.speech_pad_ms = 0;
        segs(v, m, "matched"); printf("}\n");
    }
    whisper_vad_free(v);
}
