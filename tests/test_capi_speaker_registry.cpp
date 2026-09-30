// Registry entry points that need no model: add_embedding, size, save, load,
// errors. Model independent.
#include "parakeet_capi.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__);  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

int main() {
    CHECK(parakeet_capi_abi_version() == 10);

    parakeet_speaker_registry* reg = parakeet_capi_speaker_registry_new();
    CHECK(reg != nullptr);
    const float a[4] = {1, 0, 0, 0};
    const float b[4] = {0, 1, 0, 0};
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ada", a, 4) == 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ben", b, 4) == 0);
    CHECK(parakeet_capi_speaker_registry_size(reg) == 2);
    // Same name again: averaged into the same speaker, not a third one.
    const float a2[4] = {0.9f, 0.1f, 0, 0};
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ada", a2, 4) == 0);
    CHECK(parakeet_capi_speaker_registry_size(reg) == 2);

    // Errors, each leaves the registry unchanged and says why on the registry.
    const float c3[3] = {1, 0, 0};
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "cy", c3, 3) != 0);
    CHECK(std::strstr(parakeet_capi_speaker_registry_last_error(reg), "expects") != nullptr);
    const float nan4[4] = {std::numeric_limits<float>::quiet_NaN(), 0, 0, 0};
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "dee", nan4, 4) != 0);
    const float inf4[4] = {std::numeric_limits<float>::infinity(), 0, 0, 0};
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "dee", inf4, 4) != 0);
    const float zero4[4] = {0, 0, 0, 0};
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "dee", zero4, 4) != 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "", a, 4) != 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, nullptr, a, 4) != 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ee", nullptr, 4) != 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ee", a, 0) != 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ee", a, -1) != 0);
    CHECK(parakeet_capi_speaker_registry_add_embedding(nullptr, "ee", a, 4) != 0);
    CHECK(parakeet_capi_speaker_registry_size(reg) == 2);
    // A success clears the message.
    CHECK(parakeet_capi_speaker_registry_add_embedding(reg, "ben", b, 4) == 0);
    CHECK(std::strlen(parakeet_capi_speaker_registry_last_error(reg)) == 0);

    // A registry filled this way saves and loads like any other.
    const std::filesystem::path p = std::filesystem::temp_directory_path() / "pk_test_add_embedding.bin";
    std::filesystem::remove(p);
    CHECK(parakeet_capi_speaker_registry_save(reg, p.string().c_str()) == 0);
    parakeet_speaker_registry* back = parakeet_capi_speaker_registry_load(p.string().c_str());
    CHECK(back != nullptr && parakeet_capi_speaker_registry_size(back) == 2);
    // And a loaded registry keeps its dimension: a wrong-sized vector is refused.
    if (back) CHECK(parakeet_capi_speaker_registry_add_embedding(back, "cy", c3, 3) != 0);
    parakeet_capi_speaker_registry_free(back);
    std::filesystem::remove(p);

    parakeet_capi_speaker_registry_free(reg);
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_capi_speaker_registry: PASS\n");
    return 0;
}
