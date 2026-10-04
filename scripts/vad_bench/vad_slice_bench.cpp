// load <model> N [cold] : N timed loads (ms each). speed <model> <wav> <threads> : one timed VAD run after a warm-up.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <memory>
#include "model.hpp"
#include "silero_vad.hpp"
#include "vad_json.hpp"
#include "audio_io.hpp"
#include "ggml_graph.hpp"
using clk = std::chrono::steady_clock;
struct H { std::unique_ptr<pk::Model> m; std::unique_ptr<pk::SileroVad> s; };
static H load(const std::string& p) {
    H h; std::string err;
    if (pk::gguf_is_silero(p)) h.s = pk::SileroVad::load(p, &err);
    else h.m = pk::gguf_is_vad_only(p) ? pk::Model::load_vad_only(p) : pk::Model::load(p);
    if (!h.m && !h.s) { std::fprintf(stderr, "load failed\n"); std::exit(1); }
    return h;
}
static std::string run(const H& h, const std::vector<float>& pcm) {
    pk::VadRequest r; r.kind = h.s ? pk::VadKind::kSilero : pk::VadKind::kHead; r.opts = pk::default_segmenter_opts(r.kind);
    return h.s ? pk::silero_vad_to_json(*h.s, pcm, 16000, r) : pk::vad_to_json(*h.m, pcm, r);
}
int main(int argc, char** argv) {
    std::string mode = argv[1], p = argv[2];
    if (mode == "load") {
        int n = atoi(argv[3]); bool cold = argc > 4;
        for (int i = 0; i < n; ++i) {
            if (cold) { int fd = open(p.c_str(), O_RDONLY); posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); }
            auto t0 = clk::now(); { H h = load(p); }
            std::printf("%.1f\n", std::chrono::duration<double, std::milli>(clk::now() - t0).count());
        }
    } else {
        pk::Audio a; pk::load_audio_16k_mono(argv[3], a);
        pk::set_num_threads(atoi(argv[4]));
        H h = load(p); run(h, a.samples);
        auto t0 = clk::now(); std::string o = run(h, a.samples);
        double s = std::chrono::duration<double>(clk::now() - t0).count();
        std::printf("%.4f %.1f\n", s, a.samples.size() / 16000.0 / s);
    }
}
