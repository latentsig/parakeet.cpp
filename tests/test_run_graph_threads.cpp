// A per-call n_threads in run_graph must not change the thread count of the
// process-global backend afterwards. Silero VAD asks for one thread per chunk;
// before this was scoped, a VAD pass left the ASR decode that followed on one
// thread. The Silero part runs only when PARAKEET_TEST_SILERO_GGUF is set.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "backend.hpp"
#include "ggml.h"
#include "ggml_graph.hpp"
#include "silero_vad.hpp"

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

static bool tiny_graph(int n_threads) {
    std::vector<float> out;
    const float x[4] = {1, 2, 3, 4};
    const bool ok = pk::run_graph(0, n_threads, [&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t ne[1] = {4};
        ggml_tensor* in = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne, x, sizeof(x));
        return ggml_scale(ctx, in, 2.0f);
    }, out);
    return ok && out.size() == 4 && out[3] == 8.0f;
}

int main() {
    pk::set_num_threads(0);  // no --threads
    CHECK(tiny_graph(0));
    const int before = pk::backend_thread_count();
    CHECK(before == pk::effective_threads());

    CHECK(tiny_graph(1));
    CHECK(pk::backend_thread_count() == before);
    CHECK(tiny_graph(before == 3 ? 2 : 3));
    CHECK(pk::backend_thread_count() == before);

    if (const char* gguf = std::getenv("PARAKEET_TEST_SILERO_GGUF")) {
        std::string err;
        auto vad = pk::SileroVad::load(gguf, &err);
        CHECK(vad != nullptr);
        if (vad) {
            std::vector<float> pcm(16000, 0.0f);
            CHECK(!vad->probabilities(pcm.data(), pcm.size(), 16000).empty());
            CHECK(pk::backend_thread_count() == before);
        }
    }

    // A global override still wins over the per-call count.
    pk::set_num_threads(2);
    CHECK(tiny_graph(1));
    CHECK(pk::backend_thread_count() == 2);
    pk::set_num_threads(0);

    pk::shutdown_backend();
    if (failures) return 1;
    std::printf("ok\n");
    return 0;
}
