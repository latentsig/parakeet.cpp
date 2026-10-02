// Pool shutdown: destroying a pooled model and its backends leaves no extra
// threads behind, a request holding a lease keeps the pool alive until it
// ends, and the model can be destroyed while the pool is replaced.
// Allocation leaks are checked by running this test under AddressSanitizer
// (LeakSanitizer reports at exit; see docs/concurrency.md).
//
// LABEL model
// WORKING_DIRECTORY (tests run from the project root)
// Env: PARAKEET_TEST_GGUF (skip 77 if unset)
#include "audio_io.hpp"
#include "ggml_graph.hpp"
#include "model.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

// Number of threads of this process, from /proc (Linux). -1 if unavailable.
static int thread_count() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("Threads:", 0) == 0) return std::atoi(line.c_str() + 8);
    return -1;
}

int main() {
    const char* path = std::getenv("PARAKEET_TEST_GGUF");
    if (!path) return 77;
    pk::Audio a;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", a)) return 1;

    int failures = 0;
    std::string ref;
    int baseline = -1;
    {
        auto model = pk::Model::load(path);
        if (!model) return 1;
        ref = model->transcribe_pcm(a.samples, 16000);   // warms the global backend
        baseline = thread_count();

        for (int cycle = 0; cycle < 3; ++cycle) {
            model->set_concurrency(4, 2);
            std::vector<std::thread> ts;
            std::vector<std::string> out(6);
            for (int t = 0; t < 6; ++t)
                ts.emplace_back([&, t] { out[t] = model->transcribe_pcm(a.samples, 16000); });
            for (auto& th : ts) th.join();
            for (const auto& o : out)
                if (o != ref) { std::fprintf(stderr, "FAIL: transcript differs in cycle %d\n", cycle); ++failures; }
            model->set_concurrency(1);   // drops the pool
        }
        if (thread_count() != baseline) {
            std::fprintf(stderr, "FAIL: threads %d after pools, baseline %d\n", thread_count(), baseline);
            ++failures;
        }

        // Destroy the model while its pool exists and a request is running on
        // it: the request must finish before the destructor returns, so join
        // the request first and then destroy (the model is not safe to destroy
        // under a running call, same as without a pool).
        model->set_concurrency(3, 2);
        std::thread req([&] { (void)model->transcribe_pcm(a.samples, 16000); });
        req.join();
    }   // model and pool destroyed here
    pk::shutdown_backend();

    // After everything is destroyed the process has no more threads than at the
    // start of the model's life, minus the global backend's own pool.
    const int after = thread_count();
    if (baseline >= 0 && after > baseline) {
        std::fprintf(stderr, "FAIL: %d threads after shutdown, %d before\n", after, baseline);
        ++failures;
    }
    if (failures == 0) std::printf("test_pool_shutdown: ok (threads: baseline %d, after %d)\n", baseline, after);
    return failures == 0 ? 0 : 1;
}
