// Minimal test: verify ggml_cont actually rearranges data in this backend.
#include <cstdio>
#include <vector>
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"

int main() {
    // Create a simple 2x4 tensor, transpose it, cont it, and check if data
    // is actually rearranged.
    // Original: ne=[4, 2] = [cols, rows], data: flat[c + r*4]
    //   0 1 2 3
    //   4 5 6 7
    // flat = [0, 1, 2, 3, 4, 5, 6, 7]
    //
    // After transpose: ne=[2, 4], strides say data should be
    //   0 4
    //   1 5
    //   2 6
    //   3 7
    // But data is still flat = [0, 1, 2, 3, 4, 5, 6, 7]
    //
    // After cont: data should be flat = [0, 4, 1, 5, 2, 6, 3, 7]
    // (reading row-by-row: row 0 = [0, 4], row 1 = [1, 5], etc.)

    float input_data[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<float> out;
    bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
        int64_t ne[2] = {4, 2}; // ne[0]=4 (cols), ne[1]=2 (rows)
        ggml_tensor* t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne,
                            input_data, 8 * sizeof(float));
        // t: ne=[4, 2], data=[0,1,2,3,4,5,6,7]

        // Transpose: ne=[2, 4], data unchanged
        ggml_tensor* tt = ggml_transpose(ctx, t);
        // tt: ne=[2, 4], but data is still [0,1,2,3,4,5,6,7]

        // Cont: should rearrange to [0,4,1,5,2,6,3,7]
        ggml_tensor* ct = ggml_cont(ctx, tt);
        // ct: ne=[2, 4], data should be [0,4,1,5,2,6,3,7]

        return ct;
    }, out);

    if (!ok) {
        printf("FAIL: run_graph returned false\n");
        return 1;
    }

    printf("Expected: 0 4 1 5 2 6 3 7\n");
    printf("Got:      ");
    for (int i = 0; i < 8; i++) printf("%.0f ", out[i]);
    printf("\n");

    float expected[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    bool pass = true;
    for (int i = 0; i < 8; i++) {
        if (out[i] != expected[i]) { pass = false; break; }
    }

    printf("Result: %s\n", pass ? "PASS - cont works" : "FAIL - cont is no-op");
    return pass ? 0 : 1;
}
