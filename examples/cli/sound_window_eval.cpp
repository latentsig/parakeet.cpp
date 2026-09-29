// One-off measurement: how much CED's top-1 tag changes when a clip is scored
// in short windows instead of one window of up to 10 s. For every WAV in the
// list, the reference is the top-1 class of the whole clip (<= 10 s); each
// window size W reports the share of W-second windows (hop W/2) whose top-1
// equals the reference, and the share whose top-5 contains it.
// usage: sound-window-eval <ced.gguf> <list.txt with one wav path per line>
#include "audio_io.hpp"
#include "ced_tagger.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

// Load a WAV as 16 kHz mono (pk::load_audio_16k_mono downmixes and resamples).
static bool load16k(const std::string& path, std::vector<float>& x) {
    pk::Audio a;
    if (!pk::load_audio_16k_mono(path, a)) return false;
    x = std::move(a.samples);
    return true;
}

static std::vector<int> topk(const std::vector<float>& p, int k) {
    std::vector<int> i(p.size());
    std::iota(i.begin(), i.end(), 0);
    std::partial_sort(i.begin(), i.begin() + k, i.end(), [&](int a, int b) { return p[a] > p[b]; });
    i.resize(k);
    return i;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: sound-window-eval <ced.gguf> <list.txt>\n"); return 2; }
    auto t = pk::CedTagger::load(argv[1]);
    if (!t) { std::fprintf(stderr, "load failed\n"); return 1; }
    auto score = t->scorer();
    const float sizes[] = {1.0f, 2.0f, 3.0f, 5.0f};
    std::vector<long> top1(4, 0), top5(4, 0), total(4, 0);
    std::ifstream list(argv[2]);
    std::string path;
    int clips = 0;
    while (std::getline(list, path)) {
        std::vector<float> x;
        if (path.empty() || !load16k(path, x)) continue;
        x.resize(std::min<size_t>(x.size(), 10 * 16000));
        std::vector<float> p;
        if (!score(x.data(), (int)x.size(), p)) continue;
        const int ref = topk(p, 1)[0];
        ++clips;
        for (int s = 0; s < 4; ++s) {
            const int w = (int)(sizes[s] * 16000), h = w / 2;
            for (int a = 0; a + w <= (int)x.size(); a += h) {
                if (!score(x.data() + a, w, p)) continue;
                auto k = topk(p, 5);
                top1[s] += k[0] == ref;
                top5[s] += std::find(k.begin(), k.end(), ref) != k.end();
                ++total[s];
            }
        }
    }
    std::printf("clips: %d\n| window | windows | top-1 = clip top-1 | clip top-1 in top-5 |\n|---|--:|--:|--:|\n", clips);
    for (int s = 0; s < 4; ++s)
        std::printf("| %.0f s | %ld | %.1f%% | %.1f%% |\n", sizes[s], total[s],
                    100.0 * top1[s] / std::max(1L, total[s]), 100.0 * top5[s] / std::max(1L, total[s]));
    return 0;
}
