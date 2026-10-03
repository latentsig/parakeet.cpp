// Redux with packed ternary weights (native kernel) must decode the reference
// clip to the same transcript as the dequantized Redux GGUF.
// Env: PARAKEET_TEST_GGUF_REDUX_KEEP (--ternary keep), PARAKEET_TEST_GGUF_REDUX_DEQ.
#include "model.hpp"
#include "audio_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace pk;

static const char* kReference =
    "Well, I don't wish to see it any more, observed Phoebe, turning away her eyes. "
    "It is certainly very like the old portrait.";

int main() {
    const char* keep = std::getenv("PARAKEET_TEST_GGUF_REDUX_KEEP");
    const char* deq = std::getenv("PARAKEET_TEST_GGUF_REDUX_DEQ");
    if (!keep || !deq) {
        std::puts("skip: PARAKEET_TEST_GGUF_REDUX_KEEP / _DEQ unset");
        return 77;
    }
    int failures = 0;
    std::string two_deq, two_keep;
    for (const char* path : {deq, keep}) {
        auto m = Model::load(path);
        if (!m) { std::fprintf(stderr, "FAIL: load %s\n", path); return 1; }
        const std::string got = m->transcribe_path("tests/fixtures/speech.wav", Decoder::kTDT);
        std::printf("%s\n  -> %s\n", path, got.c_str());
        if (got != kReference) { std::fprintf(stderr, "FAIL: transcript differs for %s\n", path); ++failures; }
        const std::string two = m->transcribe_path("tests/fixtures/two_speakers.wav", Decoder::kTDT);
        std::printf("  two_speakers -> %s\n", two.c_str());
        (path == keep ? two_keep : two_deq) = two;
        if (path == keep) {
            // batched == per-item on the packed model
            Audio a, b;
            if (!load_audio_16k_mono("tests/fixtures/speech.wav", a) ||
                !load_audio_16k_mono("tests/fixtures/two_speakers.wav", b)) {
                std::fprintf(stderr, "FAIL: fixture load\n");
                return 1;
            }
            const auto batch = m->transcribe_pcm_batch({a.samples, b.samples}, 16000, Decoder::kTDT);
            if (batch.size() != 2 || batch[0] != got || batch[1] != two) {
                std::fprintf(stderr, "FAIL: batched transcripts differ from per-item\n");
                for (const auto& t : batch) std::fprintf(stderr, "  batch: %s\n", t.c_str());
                ++failures;
            }
        }
    }
    if (two_deq != two_keep) {
        std::fprintf(stderr, "FAIL: two_speakers transcript differs between dequantized and packed Redux\n"
                             "  deq : %s\n  keep: %s\n", two_deq.c_str(), two_keep.c_str());
        ++failures;
    }
    if (failures) return 1;
    std::puts("test_ternary_model: OK");
    return 0;
}
