// Batched transducer decode must agree bit for bit with per-item decode.
//
// Two levels, for every model named in the environment:
//  1. step level: PredictionNet::step_batch and Joint::step_logits_batch against
//     step() and step_logits() on random inputs, for batch sizes 1, 2, 3, 4, 8,
//     16. Outputs (logits, h, c, g) are compared with memcmp.
//  2. decode level: transducer_greedy_batch against tdt_greedy / rnnt_greedy on
//     real encoder output of clips with different lengths. Token ids, frames,
//     spans and confidences are compared exactly.
// This holds on CPU only: other backends keep the ordinary batched matmul, so
// the test skips there (exit 77).
// The weight types of the decoder tensors are printed, so the log shows which
// types were covered. Models: PARAKEET_TEST_GGUF, PARAKEET_TEST_GGUF_ULTRA,
// PARAKEET_TEST_GGUF_REDUX_KEEP, PARAKEET_TEST_GGUF_REDUX_DEQ, and a colon
// separated list in PARAKEET_TEST_GGUF_LIST. Skips (77) when none is set.
#include "transducer_batch.hpp"
#include "tdt.hpp"
#include "rnnt.hpp"
#include "prediction.hpp"
#include "joint.hpp"
#include "encoder.hpp"
#include "mel.hpp"
#include "audio_io.hpp"
#include "model_loader.hpp"
#include "backend.hpp"
#include "ggml_graph.hpp"
#include "ggml.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

static const int kSizes[] = {1, 2, 3, 4, 8, 16};

static uint32_t g_rng = 12345u;
static float rnd(float scale) {
    g_rng = g_rng * 1664525u + 1013904223u;
    return ((int32_t)(g_rng >> 8) / (float)(1 << 23) - 1.0f) * scale;
}
static bool same_bits(const float* a, const float* b, size_t n) {
    return std::memcmp(a, b, n * sizeof(float)) == 0;
}

static bool check_steps(const pk::ModelLoader& ml, const pk::PredictionNet& pred,
                        const pk::Joint& joint) {
    const int H = pred.hidden_size(), L = pred.num_layers();
    const int Hj = joint.joint_hidden(), Vp = joint.V_plus();
    const int vocab_p1 = (int)ml.config().blank_id + 1;
    bool ok = true;
    for (int N : kSizes) {
        std::vector<int32_t> ids(N);
        std::vector<uint8_t> sos(N);
        pk::BatchedPredState in;
        in.h.assign(L, std::vector<float>((size_t)H * N));
        in.c.assign(L, std::vector<float>((size_t)H * N));
        for (int n = 0; n < N; ++n) {
            ids[n] = (int32_t)((uint32_t)(rnd(1.0f) * 1e6f) % (uint32_t)vocab_p1);
            sos[n] = (n % 5 == 0);
        }
        for (int l = 0; l < L; ++l)
            for (size_t i = 0; i < (size_t)H * N; ++i) {
                in.h[l][i] = rnd(0.9f);
                in.c[l][i] = rnd(2.0f);
            }
        std::vector<float> gb;
        pk::BatchedPredState ob;
        pred.step_batch(ids, sos, in, gb, ob);
        for (int n = 0; n < N; ++n) {
            pk::PredState s;
            s.h.resize(L); s.c.resize(L);
            for (int l = 0; l < L; ++l) {
                s.h[l].assign(in.h[l].begin() + (size_t)n * H, in.h[l].begin() + (size_t)(n + 1) * H);
                s.c[l].assign(in.c[l].begin() + (size_t)n * H, in.c[l].begin() + (size_t)(n + 1) * H);
            }
            std::vector<float> g1;
            pk::PredState o1;
            pred.step(ids[n], sos[n] != 0, s, g1, o1);
            bool col = same_bits(g1.data(), &gb[(size_t)n * H], H);
            for (int l = 0; l < L; ++l) {
                col = col && same_bits(o1.h[l].data(), &ob.h[l][(size_t)n * H], H);
                col = col && same_bits(o1.c[l].data(), &ob.c[l][(size_t)n * H], H);
            }
            if (!col) { std::fprintf(stderr, "  pred step N=%d column %d differs\n", N, n); ok = false; }
        }
        // Joint on the same g plus random encoder projections.
        std::vector<float> ep((size_t)Hj * N);
        for (float& v : ep) v = rnd(1.5f);
        std::vector<float> lb;
        joint.step_logits_batch(ep.data(), gb.data(), H, N, lb);
        for (int n = 0; n < N; ++n) {
            std::vector<float> l1;
            joint.step_logits(&ep[(size_t)n * Hj], &gb[(size_t)n * H], H, l1);
            if (!same_bits(l1.data(), &lb[(size_t)n * Vp], Vp)) {
                std::fprintf(stderr, "  joint step N=%d column %d differs\n", N, n);
                ok = false;
            }
        }
    }
    std::fprintf(stderr, "  step level: %s\n", ok ? "OK" : "FAIL");
    return ok;
}

struct Clip { std::vector<float> enc; int T = 0; int dm = 0; };

static bool tokens_identical(const std::vector<pk::TokenInfo>& a, const std::vector<pk::TokenInfo>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].id != b[i].id || a[i].frame != b[i].frame || a[i].span != b[i].span) return false;
        if (std::memcmp(&a[i].conf, &b[i].conf, sizeof(float)) != 0) return false;
    }
    return true;
}

static bool check_model(const char* path) {
    std::fprintf(stderr, "model %s\n", path);
    pk::ModelLoader ml;
    if (!ml.load(path)) { std::fprintf(stderr, "load failed\n"); return false; }
    const auto& cfg = ml.config();
    pk::PredictionNet pred(ml);
    pk::Joint joint(ml);

    std::set<std::string> types;
    for (const char* base : {"decoder.prediction.dec_rnn.lstm.weight_ih_l0",
                             "decoder.prediction.dec_rnn.lstm.weight_hh_l0",
                             "joint.pred.weight", "joint.joint_net.2.weight"})
        if (const ggml_tensor* t = ml.tensor(base))
            types.insert(std::string(base) + "=" + ggml_type_name(t->type));
    std::string line;
    for (const auto& s : types) line += "\n    " + s;
    std::fprintf(stderr, "  weight types:%s\n", line.c_str());

    bool ok = check_steps(ml, pred, joint);

    // Real encoder output for clips of different lengths.
    std::vector<Clip> clips;
    pk::Encoder enc(ml);
    pk::MelFrontend melfe(ml);
    for (const char* wav : {"tests/fixtures/speech.wav", "tests/fixtures/clip.wav"}) {
        pk::Audio a;
        if (!pk::load_audio_16k_mono(wav, a) || a.samples.empty()) {
            std::fprintf(stderr, "wav load failed: %s\n", wav);
            return false;
        }
        const double fr[] = {1.0, 0.75, 0.5, 0.3};
        for (double f : fr) {
            if (wav[15] == 'c' && f != 1.0) continue;  // clip.wav: full length only
            std::vector<float> pcm(a.samples.begin(), a.samples.begin() + (size_t)(a.samples.size() * f));
            std::vector<float> mel;
            int n_mels = 0, T0 = 0;
            melfe.compute(pcm, mel, n_mels, T0);
            std::vector<float> eo;
            int dm = 0, to = 0;
            enc.forward(mel, n_mels, T0, eo, dm, to);
            Clip c;
            c.T = to; c.dm = dm;
            c.enc.resize((size_t)to * dm);
            for (int t = 0; t < to; ++t)
                for (int k = 0; k < dm; ++k) c.enc[(size_t)t * dm + k] = eo[(size_t)k * to + t];
            clips.push_back(std::move(c));
        }
    }
    const bool tdt = !cfg.tdt_durations.empty();
    const int blank = (int)cfg.blank_id, maxs = (int)cfg.max_symbols;
    std::vector<std::vector<int32_t>> ref_ids(clips.size());
    std::vector<std::vector<pk::TokenInfo>> ref_tok(clips.size());
    size_t ref_tokens = 0;
    for (size_t i = 0; i < clips.size(); ++i) {
        ref_ids[i] = tdt ? pk::tdt_greedy(pred, joint, clips[i].enc, clips[i].T, clips[i].dm,
                                          cfg.tdt_durations, blank, maxs, &ref_tok[i])
                         : pk::rnnt_greedy(pred, joint, clips[i].enc, clips[i].T, clips[i].dm,
                                           blank, maxs, &ref_tok[i]);
        ref_tokens += ref_ids[i].size();
    }
    if (ref_tokens == 0) { std::fprintf(stderr, "  no tokens decoded\n"); return false; }

    bool dec_ok = true;
    for (int N : kSizes) {
        std::vector<std::vector<float>> encs(N);
        std::vector<int> Ts(N);
        std::vector<size_t> pick(N);
        for (int n = 0; n < N; ++n) {
            pick[n] = ((size_t)n * 5 + (size_t)N) % clips.size();
            encs[n] = clips[pick[n]].enc;
            Ts[n] = clips[pick[n]].T;
        }
        std::vector<std::vector<int32_t>> ids;
        std::vector<std::vector<pk::TokenInfo>> tk;
        pk::transducer_greedy_batch(pred, joint, encs, Ts, clips[0].dm,
                                    tdt ? cfg.tdt_durations : std::vector<int32_t>{},
                                    blank, maxs, ids, &tk);
        bool good = (int)ids.size() == N;
        for (int n = 0; good && n < N; ++n)
            good = ids[n] == ref_ids[pick[n]] && tokens_identical(tk[n], ref_tok[pick[n]]);
        if (!good) { std::fprintf(stderr, "  decode N=%d differs\n", N); dec_ok = false; }
    }
    std::fprintf(stderr, "  decode level: %s (%zu reference tokens over %zu clips)\n",
                 dec_ok ? "OK" : "FAIL", ref_tokens, clips.size());
    return ok && dec_ok;
}

int main() {
    std::vector<std::string> paths;
    for (const char* e : {"PARAKEET_TEST_GGUF", "PARAKEET_TEST_GGUF_ULTRA",
                          "PARAKEET_TEST_GGUF_REDUX_KEEP", "PARAKEET_TEST_GGUF_REDUX_DEQ"})
        if (const char* v = std::getenv(e)) paths.push_back(v);
    if (const char* l = std::getenv("PARAKEET_TEST_GGUF_LIST")) {
        std::string s = l;
        size_t p = 0;
        while (p <= s.size()) {
            size_t q = s.find(':', p);
            if (q == std::string::npos) q = s.size();
            if (q > p) paths.push_back(s.substr(p, q - p));
            p = q + 1;
        }
    }
    if (paths.empty()) { std::fprintf(stderr, "no model env set; skip\n"); return 77; }
    if (!pk::global_backend().is_cpu()) { std::fprintf(stderr, "not a CPU backend; skip\n"); return 77; }
    bool ok = true;
    for (const std::string& p : paths) ok = check_model(p.c_str()) && ok;
    return ok ? 0 : 1;
}
