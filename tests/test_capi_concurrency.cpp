// C-API concurrency: set_concurrency return values and the last_error contract
// under concurrent writers. Run it under ThreadSanitizer for the race check.
//
// LABEL model
// WORKING_DIRECTORY (tests run from the project root; wav path is relative)
//
// Env:
//   PARAKEET_TEST_GGUF   ASR model (skip 77 if unset)
#include "parakeet_capi.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int failures = 0;
#define CHECK(cond, ...)                                          \
    do {                                                          \
        if (!(cond)) {                                            \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                    \
            std::fprintf(stderr, "\n");                           \
            ++failures;                                           \
        }                                                         \
    } while (0)

int main() {
    const char* gguf = std::getenv("PARAKEET_TEST_GGUF");
    if (!gguf) return 77;

    // Contract of a NULL context.
    CHECK(parakeet_capi_set_concurrency(nullptr, 2, 1) == 0, "NULL ctx returns 0");

    parakeet_ctx* ctx = parakeet_capi_load(gguf);
    if (!ctx) { std::fprintf(stderr, "load failed\n"); return 1; }

    CHECK(parakeet_capi_set_concurrency(ctx, 0, 0) == 1, "0 clamps to 1");
    CHECK(parakeet_capi_set_concurrency(ctx, 3, 2) == 3, "3 backends");
    CHECK(parakeet_capi_set_concurrency(ctx, 1, 0) == 1, "back to 1 backend");

    // Two failing calls with different messages from many threads at once while
    // one thread reads the error. Every read must be one of the known messages
    // (never torn); under ThreadSanitizer there must be no race report.
    CHECK(parakeet_capi_set_concurrency(ctx, 2, 1) == 2, "2 backends");
    std::atomic<bool> stop{false};
    std::atomic<int> bad_reads{0}, reads{0};
    std::vector<std::thread> writers;
    for (int t = 0; t < 6; ++t)
        writers.emplace_back([&, t] {
            for (int i = 0; i < 400; ++i) {
                char* r = (t % 2)
                    ? parakeet_capi_transcribe_path(ctx, nullptr, 0)
                    : parakeet_capi_transcribe_pcm(ctx, nullptr, 10, 16000, 0);
                if (r) { parakeet_capi_free_string(r); ++bad_reads; }
            }
        });
    std::thread reader([&] {
        while (!stop.load()) {
            const char* e = parakeet_capi_last_error(ctx);
            const std::string s = e ? e : "";
            ++reads;
            if (s != "" && s != "wav_path is NULL" && s != "invalid samples buffer") ++bad_reads;
        }
    });
    for (auto& w : writers) w.join();
    stop = true;
    reader.join();
    CHECK(bad_reads == 0, "unexpected error text or a call that did not fail (%d)", bad_reads.load());
    const std::string last = parakeet_capi_last_error(ctx);
    CHECK(last == "wav_path is NULL" || last == "invalid samples buffer",
          "final error text: '%s'", last.c_str());
    std::printf("last_error reads: %d\n", reads.load());

    parakeet_capi_free(ctx);
    if (failures == 0) std::printf("test_capi_concurrency: ok\n");
    return failures == 0 ? 0 : 1;
}
