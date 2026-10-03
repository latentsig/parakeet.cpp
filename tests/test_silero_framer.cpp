// Silero VAD chunking, context carry and reflect padding. Needs no model file.
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>

#include "silero_vad.hpp"

using pk::silero::Framer;
using pk::silero::RateParams;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

static std::vector<float> ramp(size_t n, float start = 1.0f) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = start + (float)i;
    return v;
}

// All inputs a Framer yields for `pcm` fed in pieces of the given sizes, then flushed.
static std::vector<std::vector<float>> run(const RateParams& p, const std::vector<float>& pcm,
                                           const std::vector<size_t>& pieces, bool flush) {
    Framer f(p);
    std::vector<std::vector<float>> out;
    std::vector<float> in((size_t)p.input_len());
    size_t pos = 0, k = 0;
    while (pos < pcm.size()) {
        const size_t n = std::min(pieces[k++ % pieces.size()], pcm.size() - pos);
        f.push(pcm.data() + pos, n);
        pos += n;
        while (f.next(in.data())) out.push_back(in);
    }
    if (flush && f.next_padded(in.data())) out.push_back(in);
    return out;
}

int main() {
    // Rates and geometry.
    CHECK(pk::silero::rate_params(16000) && pk::silero::rate_params(8000));
    CHECK(!pk::silero::rate_params(44100) && !pk::silero::rate_params(0));
    for (int sr : {16000, 8000}) {
        const RateParams& p = *pk::silero::rate_params(sr);
        CHECK(p.sample_rate == sr);
        CHECK((double)p.chunk / sr == pk::kSileroFrameSec);  // 32 ms at both rates
        CHECK(p.context * 8 == p.chunk && p.rpad == p.context);
        // The STFT yields exactly 4 frames, which the encoder reduces to 1.
        CHECK((p.input_len() - p.n_fft) / p.hop + 1 == 4);
    }
    CHECK(pk::kSileroFrameMs == 32);

    // Reflect padding without the edge sample: [.. a b c d e] + [d c b a] for rpad 4 on 5 samples.
    {
        float x[9] = {1, 2, 3, 4, 5, 0, 0, 0, 0};
        pk::silero::reflect_pad_right(x, 5, 4);
        const float want[9] = {1, 2, 3, 4, 5, 4, 3, 2, 1};
        for (int i = 0; i < 9; ++i) CHECK(x[i] == want[i]);
    }

    for (int sr : {16000, 8000}) {
        const RateParams& p = *pk::silero::rate_params(sr);
        const size_t C = (size_t)p.chunk, X = (size_t)p.context;
        const std::vector<float> pcm = ramp(C * 3 + 10);

        // Nothing is produced before a full chunk; pending counts the buffer.
        {
            Framer f(p);
            std::vector<float> in((size_t)p.input_len());
            f.push(pcm.data(), C - 1);
            CHECK(!f.next(in.data()) && f.pending() == C - 1);
            f.push(pcm.data() + C - 1, 1);
            CHECK(f.next(in.data()) && f.pending() == 0);
            // First input: zero context, then the chunk, then the reflected tail.
            for (size_t i = 0; i < X; ++i) CHECK(in[i] == 0.0f);
            for (size_t i = 0; i < C; ++i) CHECK(in[X + i] == pcm[i]);
            for (int i = 0; i < p.rpad; ++i) CHECK(in[X + C + (size_t)i] == pcm[C - 2 - (size_t)i]);
            CHECK(!f.next_padded(in.data()));  // nothing pending: no padded chunk
        }

        // Context carry: the second input starts with the last `context` samples of the first chunk.
        {
            const auto in = run(p, pcm, {C}, false);
            CHECK(in.size() == 3);
            for (size_t i = 0; i < X; ++i) CHECK(in[1][i] == pcm[C - X + i]);
            for (size_t i = 0; i < X; ++i) CHECK(in[2][i] == pcm[2 * C - X + i]);
            for (size_t i = 0; i < C; ++i) CHECK(in[2][X + i] == pcm[2 * C + i]);
        }

        // Any split of the input gives the same inputs, bit for bit, and a partial
        // last chunk is zero-padded by the flush.
        const auto whole = run(p, pcm, {pcm.size()}, true);
        CHECK(whole.size() == 4);
        {
            const size_t X0 = X;
            // Last input: context | 10 samples | zeros | reflect of the zero-padded chunk.
            const auto& last = whole[3];
            for (size_t i = 0; i < 10; ++i) CHECK(last[X0 + i] == pcm[3 * C + i]);
            for (size_t i = 10; i < C; ++i) CHECK(last[X0 + i] == 0.0f);
            for (int i = 0; i < p.rpad; ++i) CHECK(last[X0 + C + (size_t)i] == last[X0 + C - 2 - (size_t)i]);
        }
        std::mt19937 rng(7);
        for (int trial = 0; trial < 20; ++trial) {
            std::vector<size_t> pieces;
            for (int k = 0; k < 8; ++k) pieces.push_back(rng() % (2 * C) + (trial % 2 ? 0 : 1));
            if (pieces[0] == 0) pieces[0] = 1;
            bool progress = false;
            for (size_t q : pieces) progress |= q > 0;
            if (!progress) continue;
            CHECK(run(p, pcm, pieces, true) == whole);
        }
        CHECK(run(p, pcm, {1}, true) == whole);

        // reset() starts again from a zero context.
        Framer f(p);
        std::vector<float> in((size_t)p.input_len());
        f.push(pcm.data(), C + 5);
        CHECK(f.next(in.data()));
        f.reset();
        CHECK(f.pending() == 0);
        f.push(pcm.data(), C);
        CHECK(f.next(in.data()));
        CHECK(in == whole[0]);
    }

    if (failures) return 1;
    std::printf("test_silero_framer OK\n");
    return 0;
}
