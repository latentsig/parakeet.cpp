# Concurrent requests

By default one loaded model runs one request at a time. Calls from several
threads are safe, but they queue behind one compute backend. The backend pool
lets several requests run at once against one loaded model. It is off by
default and changes nothing until you turn it on.

## How it works

- A pool holds K CPU backends. Each has its own ggml thread team, graph
  allocator and mutex.
- A request borrows one backend for its whole duration. The library starts no
  threads of its own: the request runs on the caller's thread and the backend
  only adds its compute threads. When all K backends are busy, the next caller
  waits.
- Weights are loaded once and shared by every backend (read only). The packed
  ternary weights and the decoder objects are shared the same way. A pool costs
  graph buffers, not weight copies.
- Results do not change. The encoder output does not depend on the thread
  count (measured bitwise equal at 1, 2, 4, 6 and 8 threads), and decode is
  per row at one column, so a backend with fewer threads produces the same
  floats. `tests/test_pool_stress.cpp` checks text, token ids, frames, spans,
  confidences and word timestamps for equality against the no-pool result.
- K = 1 (the default) removes the pool and uses the process-wide backend
  exactly as before.

## Turn it on

C-API (additive, the ABI version is unchanged):

```c
parakeet_ctx *ctx = parakeet_capi_load("model.gguf");
int k = parakeet_capi_set_concurrency(ctx, /*backends*/ 4, /*threads_each*/ 2);
// k is the effective number of backends. Now call parakeet_capi_transcribe_*
// from up to 4 threads at once on the same ctx.
```

C++: `pk::Model::set_concurrency(backends, threads_each)`.

CLI and server:

```sh
parakeet-cli bench --model m.gguf --manifest clips.txt --concurrency 4 --threads 2
parakeet-server --model m.gguf --concurrency 4 --threads 2
```

With `--concurrency K` greater than 1, `--threads` is the thread count of each
backend. If you leave it out, the default budget of 8 threads is divided by K
(at least 1 each). `bench` then adds `concurrency`, `wall_ms` and
`aggregate_rtfx` to its JSON. The server runs transcriptions in parallel; the
sound tagger, which is not thread safe, keeps a lock.

Call `set_concurrency` while no request is running, for example right after
load. A call that is already running finishes on the old pool.

## Choosing K and the thread count

- Keep `backends * threads_each` at or below the number of physical cores. The
  compute thread teams spin, and an oversubscribed machine slows every
  request. The library does not pin threads. Pin worker processes to disjoint
  cores yourself if you need that (for example `taskset`).
- Per-request latency gets somewhat worse, because each request has fewer
  threads. Any gain is in aggregate throughput, and it depends on model size
  and core count (see "Measured throughput"). Leave the pool off when you
  serve one request at a time.
- Try K greater than 1 only when `backends * threads_each` fits the physical
  cores. Measure your own model and clips before you enable it, and compare
  against one backend with all the threads. K = 1 stays the default because
  the pool is not always faster.
- Each backend keeps its own graph buffers, and they grow to the largest graph
  it has run. On the 110M Q8_0 model, two backends that had run 23 s clips and
  a batch of six clips held about 1.3 GB together, and four held about 2.5 GB.
  `Model::pool_working_set_bytes()` reports the current sum. Size K with it.
- A GPU device keeps one backend and the process-wide lock.
  `set_concurrency` then returns 1.
- Streaming sessions and diarization are not pooled. They keep their current
  single-call semantics.

## last_error with several threads

`parakeet_capi_last_error(ctx)` returns the message of the most recently
finished call on that context, from any thread. Writes are locked. A
successful call clears it, so under concurrency a success on one thread can
clear a failure from another. The pointer stays valid until the next
`parakeet_capi_last_error` call on that context or until the context is freed.
Copy the text if you need it longer. If you need the exact error of your own
call, use one context per thread (this duplicates the packed weights per
context, so prefer the pool when memory matters).

## Measured throughput

Aggregate throughput over 16 clips (8 times a 7.4 s clip and 8 times a 23.6 s
two-speaker clip, 248 s of audio) with 8 cores pinned, interleaved A/B runs,
5 runs per configuration. RTFx is audio seconds over wall seconds, higher is
faster.

110M Q8_0 (`tdt_ctc-110m-q8_0`), cores 0 to 7 pinned, aggregate RTFx over 5
interleaved runs. "base" is the binary built before the pool, "new K1" the same
default after it, so those two rows check that the default did not move.

| Configuration | Min | Median | Max | Median vs new K1 |
|---|---|---|---|---|
| base binary, 1 backend, 8 threads | 112.3 | 117.1 | 124.7 | n/a |
| new binary, 1 backend, 8 threads | 109.7 | 119.4 | 122.6 | 1.00x |
| 2 backends, 4 threads each | 131.8 | 145.9 | 149.0 | 1.22x |
| 4 backends, 2 threads each | 147.5 | 151.1 | 153.0 | 1.27x |

Load average at the start of each run: 12.1 to 13.0. The transcripts of every
configuration were equal to the base binary's.

Larger models behave differently. On 0.6B-class models (packed Redux and Ultra
Q8_0) with 8 total threads, the pool was slower than one backend with 8
threads. Median ratios against one backend, measured on two different hosts:

| Configuration | Ratio vs 1 backend, 8 threads |
|---|---|
| K = 2 | 0.62x to 0.87x |
| K = 4 | 0.73x to 0.91x |

With 16 threads on a 16-core host, packed Redux gave 1.08x at K = 2 and 1.30x
at K = 4. So the benefit depends on model size and core count: it appeared on
the 110M model and on a 16-core host, and not on 0.6B models with 8 threads.

Quiet-machine numbers are missing. All of these runs were on loaded machines
(the 110M runs had a load average of 12 to 13), so read every ratio as
indicative and re-measure on an idle host before you rely on it. The
K = 2 and K = 4 minimums are above the K = 1 maximum, and the two K = 1 rows
overlap, so the default did not move. An attempt on the 0.6B F16 model was
abandoned: other jobs pushed the load above 40 and the numbers were useless.

## Tests

- `test_backend_pool`: routing, lease limits, reentrancy, waiting callers,
  shutdown. No model needed.
- `test_capi_concurrency`: `set_concurrency` return values and concurrent
  `last_error` writes with a reader.
- `test_pool_stress`: N threads over K backends, results equal to the no-pool
  result. `PARAKEET_STRESS_QUICK=1` shortens it for slow builds.
- `test_pool_shutdown`: pools leave no extra threads; results stay equal over
  create and destroy cycles.

ThreadSanitizer needs a build without OpenMP (libgomp is not instrumented) and,
on kernels with high-entropy address randomization, ASLR switched off for the
run:

```sh
cmake -B build-tsan -DPARAKEET_BUILD_TESTS=ON -DGGML_OPENMP=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS="-fsanitize=thread -O1 -g" -DCMAKE_CXX_FLAGS="-fsanitize=thread -O1 -g" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
cmake --build build-tsan -j --target test_backend_pool test_capi_concurrency test_pool_stress
export PARAKEET_TEST_GGUF=model.gguf PARAKEET_STRESS_QUICK=1
setarch x86_64 -R build-tsan/tests/test_pool_stress
```

For leaks, build with `-fsanitize=address` and run `test_pool_shutdown`.

## Shared state that was audited

- The process-wide backend: its lazy creation and thread-count sync now run
  under its mutex, because pooled requests still ask it for the device name.
- The loader's weight buffer: an atomic pointer, set last, with the one-time
  setup serialized.
- `PredictionNet` and `Joint`: built once per model; the lazily filled
  embedding table is filled under a `std::once_flag`.
- The graph allocation size counter used by tests: atomic.
- The ternary store lookup (locked), the FFT plan cache (thread local) and the
  mel front end (built per call): already safe.
