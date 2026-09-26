// Test: subpixel reshape WITHOUT 3D permute
// Uses reshape_3d → reshape_2d → 2D transpose+cont (avoids 3D cont bug)
#include <cstdio>
#include <vector>
#include <cstring>
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"

int main() {
    // Small test: T_enc=4, up=2, tf=3
    // conv_out has ne=[T_enc, tf*up] = [4, 6], data: flat[t + c*4]
    // c = h*up + u
    //
    // Reference subpixel:
    //   x.view(B, C//up, up, T) → x.view(B, C//up, up*T)
    //   up_pk[h, t'] = conv[h*up+u, t] where t' = u*T + t
    //
    // In ggml (column-major):
    // 1. reshape_3d(T_enc, up, tf): ne=[T_enc, up, tf], element(t,u,h) = flat[t + u*T + h*up*T]
    //    = flat[t + (h*up+u)*T] = flat[t + c*T] ✓
    // 2. reshape_2d(T_out, tf): ne=[T_out, tf], element(t',h) = flat[t' + h*T_out]
    //    t' = t + u*T → flat[t + u*T + h*up*T] = flat[t + c*T] ✓
    // 3. transpose: ne=[tf, T_out], element(h,t') = flat[t' + h*T_out] (view, strides change)
    // 4. cont: copies to flat[h + t'*tf] (2D cont, which WORKS)

    const int T_enc = 4, up = 2, tf = 3;
    const int T_out = T_enc * up;

    float conv_data[24];
    for (int c = 0; c < tf * up; c++)
        for (int t = 0; t < T_enc; t++)
            conv_data[t + c * T_enc] = (c + 1) * 10 + t;

    float bias_data[6] = {0};

    std::vector<float> out;
    bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
        int64_t ne[2] = {T_enc, tf * up};
        ggml_tensor* conv_out = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, ne,
                                conv_data, 24 * sizeof(float));

        int64_t bne[2] = {1, tf * up};
        ggml_tensor* bias = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 2, bne,
                               bias_data, 6 * sizeof(float));
        conv_out = ggml_add(ctx, conv_out, bias);

        // Subpixel: reshape_3d → reshape_2d → transpose → cont
        ggml_tensor* upsampled = ggml_reshape_3d(ctx, conv_out, T_enc, up, tf);
        upsampled = ggml_reshape_2d(ctx, upsampled, T_enc * up, tf);  // ne=[T_out, tf]
        upsampled = ggml_cont(ctx, ggml_transpose(ctx, upsampled));   // ne=[tf, T_out]
        return upsampled;
    }, out);

    if (!ok) { printf("FAIL: run_graph returned false\n"); return 1; }

    // out has ne=[tf, T_out], data: flat[h + t'*tf]
    printf("Expected:\n");
    for (int h = 0; h < tf; h++) {
        printf("  h%d: ", h);
        for (int tp = 0; tp < T_out; tp++) {
            int u = tp / T_enc;
            int t = tp % T_enc;
            int c = h * up + u;
            printf("%.0f ", conv_data[t + c * T_enc]);
        }
        printf("\n");
    }
    printf("Got:\n");
    for (int h = 0; h < tf; h++) {
        printf("  h%d: ", h);
        for (int tp = 0; tp < T_out; tp++) {
            printf("%.0f ", out[h + tp * tf]);
        }
        printf("\n");
    }

    bool pass = true;
    for (int h = 0; h < tf; h++)
        for (int tp = 0; tp < T_out; tp++) {
            int u = tp / T_enc;
            int t = tp % T_enc;
            int c = h * up + u;
            if (out[h + tp * tf] != conv_data[t + c * T_enc]) { pass = false; }
        }
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
