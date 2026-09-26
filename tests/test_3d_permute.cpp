// Test: permute + cont on a 3D tensor (the subpixel case)
#include <cstdio>
#include <vector>
#include <cstring>
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"

int main() {
    // 3D tensor: ne=[4, 2, 3] = [T, up, tf]
    // Data: flat[t + u*4 + h*8]
    // h=0: [10,11,12,13, 20,21,22,23]
    // h=1: [30,31,32,33, 40,41,42,43]
    // h=2: [50,51,52,53, 60,61,62,63]
    //
    // After permute(2,0,1,3): ne=[3, 4, 2] = [tf, T, up]
    // element(h, t, u) = old element(t, u, h) = flat[t + u*4 + h*8]
    //
    // After cont: data should be contiguous
    // new_flat[h + t*3 + u*12] = old_flat[t + u*4 + h*8]
    //
    // After reshape_2d(3, 8): ne=[3, 8] = [tf, T_out]
    // element(h, t') = new_flat[h + t'*3]
    // where t' = t + u*4 = t + u*T_enc
    //
    // Expected:
    // h=0: t'=0: t=0,u=0 → flat[0+0*4+0*8] = 10
    //      t'=1: t=1,u=0 → flat[1+0+0] = 11
    //      t'=2: t=2,u=0 → flat[2] = 12
    //      t'=3: t=3,u=0 → flat[3] = 13
    //      t'=4: t=0,u=1 → flat[0+4+0] = 20
    //      t'=5: t=1,u=1 → flat[1+4] = 21
    //      t'=6: t=2,u=1 → flat[2+4] = 22
    //      t'=7: t=3,u=1 → flat[3+4] = 23

    const int T_enc = 4, up = 2, tf = 3;
    float data[24];
    for (int h = 0; h < tf; h++)
        for (int u = 0; u < up; u++)
            for (int t = 0; t < T_enc; t++)
                data[t + u*T_enc + h*up*T_enc] = (h*up + u + 1) * 10 + t;

    printf("Input data: ");
    for (int i = 0; i < 24; i++) printf("%.0f ", data[i]);
    printf("\n\n");

    // Test 1: permute + cont + reshape_2d
    {
        std::vector<float> out;
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t ne[3] = {T_enc, up, tf};
            ggml_tensor* t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, ne,
                                data, 24 * sizeof(float));
            // t: ne=[T_enc, up, tf] = [4, 2, 3]
            ggml_tensor* p = ggml_permute(ctx, t, 2, 0, 1, 3);
            // p: ne=[tf, T_enc, up] = [3, 4, 2]
            ggml_tensor* c = ggml_cont(ctx, p);
            // c: ne=[3, 4, 2], contiguous
            ggml_tensor* r = ggml_reshape_2d(ctx, c, tf, T_enc * up);
            // r: ne=[tf, T_out] = [3, 8]
            return r;
        }, out);

        printf("Test 1 (permute+cont+reshape_2d):\n");
        printf("Expected:\n");
        for (int h = 0; h < tf; h++) {
            printf("  h%d: ", h);
            for (int tp = 0; tp < T_enc * up; tp++) {
                int u = tp / T_enc;
                int t = tp % T_enc;
                printf("%.0f ", data[t + u*T_enc + h*up*T_enc]);
            }
            printf("\n");
        }
        printf("Got (out[h + t'*tf]):\n");
        for (int h = 0; h < tf; h++) {
            printf("  h%d: ", h);
            for (int tp = 0; tp < T_enc * up; tp++) {
                printf("%.0f ", out[h + tp * tf]);
            }
            printf("\n");
        }

        bool pass = true;
        for (int h = 0; h < tf; h++) {
            for (int tp = 0; tp < T_enc * up; tp++) {
                int u = tp / T_enc;
                int t = tp % T_enc;
                float expected = data[t + u*T_enc + h*up*T_enc];
                if (out[h + tp * tf] != expected) { pass = false; break; }
            }
        }
        printf("Result: %s\n\n", pass ? "PASS" : "FAIL");
    }

    // Test 2: Just cont (no permute) — should be identity copy
    {
        std::vector<float> out;
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t ne[3] = {T_enc, up, tf};
            ggml_tensor* t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, ne,
                                data, 24 * sizeof(float));
            return ggml_cont(ctx, t);
        }, out);

        printf("Test 2 (cont identity):\n");
        printf("Expected: ");
        for (int i = 0; i < 24; i++) printf("%.0f ", data[i]);
        printf("\nGot:      ");
        for (int i = 0; i < 24; i++) printf("%.0f ", out[i]);
        printf("\nResult: %s\n\n", data == out.data() ? "?" :
            (memcmp(data, out.data(), 24*sizeof(float)) == 0 ? "PASS" : "FAIL"));
    }

    // Test 3: permute + cont only (no reshape)
    {
        std::vector<float> out;
        bool ok = pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
            int64_t ne[3] = {T_enc, up, tf};
            ggml_tensor* t = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 3, ne,
                                data, 24 * sizeof(float));
            ggml_tensor* p = ggml_permute(ctx, t, 2, 0, 1, 3);
            ggml_tensor* c = ggml_cont(ctx, p);
            return c;
        }, out);

        // After permute(2,0,1,3) + cont: ne=[3, 4, 2]
        // element(h, t, u) = old_element(t, u, h) = data[t + u*4 + h*8]
        // Contiguous: out[h + t*3 + u*12]
        printf("Test 3 (permute+cont, no reshape):\n");
        printf("Expected (h + t*3 + u*12):\n");
        for (int h = 0; h < tf; h++)
            for (int u = 0; u < up; u++)
                for (int t = 0; t < T_enc; t++) {
                    int idx = h + t*3 + u*12;
                    printf("  [%d] = %.0f (h=%d,t=%d,u=%d)\n", idx,
                           data[t + u*4 + h*8], h, t, u);
                }
        printf("Got:\n");
        for (int i = 0; i < 24; i++) {
            // Find which (h,t,u) this should be
            // i = h + t*3 + u*12 → h = i%3, t = (i/3)%4, u = i/12
            int h = i % 3, t = (i / 3) % 4, u = i / 12;
            printf("  [%d] = %.0f (should be h=%d,t=%d,u=%d → %.0f)\n",
                   i, out[i], h, t, u, data[t + u*4 + h*8]);
        }

        bool pass = true;
        for (int i = 0; i < 24; i++) {
            int h = i % 3, t = (i / 3) % 4, u = i / 12;
            if (out[i] != data[t + u*4 + h*8]) { pass = false; }
        }
        printf("Result: %s\n", pass ? "PASS" : "FAIL");
    }

    return 0;
}
