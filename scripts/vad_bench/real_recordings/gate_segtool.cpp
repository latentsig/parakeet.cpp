// Reads raw float32 probabilities and prints the regions of speech_regions or segment_by_vad, for verify_gate.py.
// usage: gate_segtool probs.f32 total_sec kind(head|silero) run_gate mode(speech|segments)
#include "vad_segmenter.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
int main(int argc, char** argv) {
    std::FILE* f = std::fopen(argv[1], "rb"); std::vector<float> p; float v; while (std::fread(&v, 4, 1, f) == 1) p.push_back(v); std::fclose(f);
    double total = std::atof(argv[2]);
    pk::SegmenterOpts o = pk::default_segmenter_opts(!std::strcmp(argv[3], "silero") ? pk::VadKind::kSilero : pk::VadKind::kHead);
    o.run_gate = (float)std::atof(argv[4]);
    auto r = !std::strcmp(argv[5], "speech") ? pk::speech_regions(p, total, o) : pk::segment_by_vad(p, total, o);
    for (auto& s : r) std::printf("%.6f %.6f\n", s.start, s.end);
}
