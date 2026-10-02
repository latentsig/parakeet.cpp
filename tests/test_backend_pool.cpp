// Backend pool (B2a): thread-local routing, lease limits, reentrancy, shutdown.
// Model independent. Graphs are tiny element-wise adds, so the test checks the
// routing and bookkeeping, not model output.
#include "backend.hpp"
#include "backend_pool.hpp"
#include "ggml.h"
#include "ggml_graph.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>
#include <vector>

static int failures = 0;
#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, msg); \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

// out[i] = a[i] + b[i]; returns true and fills `out` on success.
static bool run_add(const std::vector<float>& a, const std::vector<float>& b,
                    std::vector<float>& out) {
    return pk::run_graph(0, 0, [&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t ne[1] = {(int64_t)a.size()};
        ggml_tensor* ta = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne, a.data(),
                                                 a.size() * sizeof(float));
        ggml_tensor* tb = pk::graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne, b.data(),
                                                 b.size() * sizeof(float));
        return ggml_add(ctx, ta, tb);
    }, out);
}

static bool add_ok(float x) {
    std::vector<float> a(64, x), b(64, 1.0f), out;
    if (!run_add(a, b, out) || out.size() != 64) return false;
    for (float v : out) if (v != x + 1.0f) return false;
    return true;
}

static void test_routing() {
    // Without a lease the global backend runs the graph and no pool slot is hit.
    auto pool = std::make_shared<pk::BackendPool>(2, 1);
    CHECK(pool->size() == 2, "pool size");
    CHECK(pool->threads_each() == 1, "threads_each");
    CHECK(pk::current_route() == nullptr, "no route before a lease");
    CHECK(add_ok(1.0f), "global add");
    CHECK(pool->total_runs() == 0, "global run does not touch the pool");
    {
        pk::PoolLease lease(pool);
        CHECK(lease.active(), "lease active");
        CHECK(pk::current_route() != nullptr, "route set under a lease");
        CHECK(pk::current_route()->backend.n_threads() == 1, "pool backend thread count");
        CHECK(add_ok(2.0f), "pooled add");
        CHECK(add_ok(3.0f), "pooled add again");
        CHECK(pool->in_use() == 1, "one backend in use");
    }
    CHECK(pk::current_route() == nullptr, "route cleared after the lease");
    CHECK(pool->in_use() == 0, "backend returned");
    CHECK(pool->total_runs() == 2, "two runs went through the pool");
    CHECK(add_ok(4.0f), "global add after the lease");
    CHECK(pool->total_runs() == 2, "global run after the lease does not touch the pool");
    CHECK(pool->working_set_bytes() > 0, "working set reported after use");
}

static void test_reentrant_and_null() {
    auto pool = std::make_shared<pk::BackendPool>(1, 1);
    pk::PoolLease outer(pool);
    {
        // A nested lease on the same thread must not wait for a second backend.
        pk::PoolLease inner(pool);
        CHECK(!inner.active(), "nested lease is inert");
        CHECK(add_ok(5.0f), "add under nested lease");
    }
    CHECK(pk::current_route() != nullptr, "outer route survives the inner lease");
    pk::PoolLease none(nullptr);
    CHECK(!none.active(), "null pool lease is inert");
}

static void test_limit_and_peak() {
    // 6 threads, 2 backends: never more than 2 in use at once, all runs correct.
    auto pool = std::make_shared<pk::BackendPool>(2, 1);
    std::atomic<int> bad{0};
    std::vector<std::thread> ts;
    for (int t = 0; t < 6; ++t)
        ts.emplace_back([&, t] {
            for (int i = 0; i < 20; ++i) {
                pk::PoolLease lease(pool);
                if (pool->in_use() > 2) ++bad;
                if (!add_ok((float)(t * 100 + i))) ++bad;
            }
        });
    for (auto& th : ts) th.join();
    CHECK(bad == 0, "concurrent pooled runs correct and bounded");
    CHECK(pool->peak_in_use() <= 2, "peak in use within K");
    CHECK(pool->peak_in_use() >= 1, "peak in use recorded");
    CHECK(pool->total_runs() == 120, "every run went through the pool");
    CHECK(pool->in_use() == 0, "all backends returned");
}

static void test_waiter_unblocks() {
    auto pool = std::make_shared<pk::BackendPool>(1, 1);
    std::atomic<bool> second_ran{false};
    std::thread first_holder;
    {
        auto lease = std::make_unique<pk::PoolLease>(pool);
        std::thread second([&] {
            pk::PoolLease l(pool);  // blocks until the first lease is released
            second_ran = true;
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        CHECK(!second_ran, "second caller waits while the only backend is leased");
        lease.reset();
        second.join();
    }
    CHECK(second_ran, "second caller runs after release");
}

static void test_shutdown_waits() {
    // Destroying the last owner while a lease is out must wait for the lease:
    // the lease keeps the pool alive, and the pool frees all backends on exit.
    std::weak_ptr<pk::BackendPool> weak;
    {
        auto pool = std::make_shared<pk::BackendPool>(3, 1);
        weak = pool;
        auto lease = std::make_unique<pk::PoolLease>(pool);
        pool.reset();
        CHECK(!weak.expired(), "lease keeps the pool alive");
        CHECK(add_ok(7.0f), "run after the owner dropped the pool");
        lease.reset();
    }
    CHECK(weak.expired(), "pool destroyed after the last lease");
    // Many create/destroy cycles must not leak threads or hang.
    for (int i = 0; i < 20; ++i) {
        auto p = std::make_shared<pk::BackendPool>(2, 2);
        pk::PoolLease l(p);
        if (!add_ok(1.0f)) ++failures;
    }
}

int main() {
    test_routing();
    test_reentrant_and_null();
    test_limit_and_peak();
    test_waiter_unblocks();
    test_shutdown_waits();
    pk::shutdown_backend();
    if (failures == 0) std::printf("test_backend_pool: ok\n");
    return failures == 0 ? 0 : 1;
}
