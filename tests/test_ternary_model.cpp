// Redux with packed ternary weights (native kernel) must decode the reference
// clip to the same transcript as the dequantized Redux GGUF.
// Env: PARAKEET_TEST_GGUF_REDUX_KEEP (--ternary keep), PARAKEET_TEST_GGUF_REDUX_DEQ.
#include "model.hpp"

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
    for (const char* path : {deq, keep}) {
        auto m = Model::load(path);
        if (!m) { std::fprintf(stderr, "FAIL: load %s\n", path); return 1; }
        const std::string got = m->transcribe_path("tests/fixtures/speech.wav", Decoder::kTDT);
        std::printf("%s\n  -> %s\n", path, got.c_str());
        if (got != kReference) { std::fprintf(stderr, "FAIL: transcript differs for %s\n", path); ++failures; }
    }
    if (failures) return 1;
    std::puts("test_ternary_model: OK");
    return 0;
}
