// Standalone diarize tool — loads a diarization GGUF and diarizes a WAV.
// Usage: diarize <gguf> <wav>
// Prints JSON segments to stdout.
#include "parakeet_capi.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <gguf> <wav>\n", argv[0]);
        return 1;
    }
    parakeet_ctx* ctx = parakeet_capi_load(argv[1]);
    if (!ctx) {
        fprintf(stderr, "failed to load %s\n", argv[1]);
        return 1;
    }
    char* json = parakeet_capi_diarize_path(ctx, argv[2]);
    if (!json) {
        fprintf(stderr, "diarize failed: %s\n", parakeet_capi_last_error(ctx));
        parakeet_capi_free(ctx);
        return 1;
    }
    printf("%s\n", json);
    parakeet_capi_free_string(json);
    parakeet_capi_free(ctx);
    return 0;
}
