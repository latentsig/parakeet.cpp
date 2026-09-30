// The transducer decoder objects (PredictionNet, Joint) are built once per Model
// and reused across utterances. Reusing them must not change any result:
//  - two consecutive transcribes of the same clip agree exactly (text, token
//    ids, frames, spans and confidences),
//  - a fresh Model (fresh decoder objects) agrees exactly with the reused one,
//  - transcribe_pcm_batch agrees on text and token ids/frames (batched matmuls
//    are only close in float, so confidences are compared with a tolerance),
//  - two threads decoding on two different Model instances both work.
// Models: PARAKEET_TEST_GGUF_REDUX_KEEP and/or PARAKEET_TEST_GGUF_ULTRA.
// Skips (77) when neither is set.
#include "model.hpp"
#include "audio_io.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

// Batched encode (padded, tiled matmuls) is only close to the per-clip encode in
// float, so confidences move by about 0.01 on the packed ternary model; token
// ids, frames and spans are compared exactly.
static const float kBatchConfTol = 0.05f;

static bool same_tokens(const std::vector<pk::TokenInfo>& a,
                        const std::vector<pk::TokenInfo>& b, float conf_tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].id != b[i].id || a[i].frame != b[i].frame || a[i].span != b[i].span)
            return false;
        if (conf_tol == 0.0f ? a[i].conf != b[i].conf
                             : std::fabs(a[i].conf - b[i].conf) > conf_tol)
            return false;
    }
    return true;
}

static bool check_model(const char* path) {
    std::fprintf(stderr, "model %s\n", path);
    auto model = pk::Model::load(path);
    auto model2 = pk::Model::load(path);
    if (!model || !model2) { std::fprintf(stderr, "load failed\n"); return false; }
    pk::Audio a;
    if (!pk::load_audio_16k_mono("tests/fixtures/speech.wav", a) || a.samples.empty()) {
        std::fprintf(stderr, "wav load failed\n"); return false;
    }
    std::vector<float> half(a.samples.begin(), a.samples.begin() + (a.samples.size() * 3) / 4);

    // A fresh Model gives the reference (its decoder objects are used once).
    pk::Transcription ref  = model2->transcribe_with_timestamps(a.samples, 16000);
    pk::Transcription ref2 = model2->transcribe_with_timestamps(half, 16000);
    bool ok = !ref.tokens.empty();

    // Reused decoder objects: same clip twice, interleaved with another clip.
    pk::Transcription r1 = model->transcribe_with_timestamps(a.samples, 16000);
    pk::Transcription r2 = model->transcribe_with_timestamps(half, 16000);
    pk::Transcription r3 = model->transcribe_with_timestamps(a.samples, 16000);
    ok = ok && r1.text == ref.text && same_tokens(r1.tokens, ref.tokens, 0.0f);
    ok = ok && r3.text == ref.text && same_tokens(r3.tokens, ref.tokens, 0.0f);
    ok = ok && r2.text == ref2.text && same_tokens(r2.tokens, ref2.tokens, 0.0f);
    std::fprintf(stderr, "  repeat/fresh: %s (%zu tokens)\n", ok ? "OK" : "FAIL", ref.tokens.size());

    // The plain-text entry point shares the same objects.
    ok = ok && model->transcribe_pcm(a.samples, 16000) == ref.text;

    // Batched path with the same reused objects.
    auto b = model->transcribe_pcm_batch_with_timestamps({a.samples, half}, 16000);
    if (b.size() != 2) { std::fprintf(stderr, "  batch size %zu\n", b.size()); return false; }
    bool bok = b[0].text == ref.text && b[1].text == ref2.text
            && same_tokens(b[0].tokens, ref.tokens, kBatchConfTol)
            && same_tokens(b[1].tokens, ref2.tokens, kBatchConfTol);
    auto bt = model->transcribe_pcm_batch({a.samples, half}, 16000);
    bok = bok && bt.size() == 2 && bt[0] == ref.text && bt[1] == ref2.text;
    if (!bok) {
        for (int i = 0; i < 2; ++i) {
            const pk::Transcription& r = i ? ref2 : ref;
            std::fprintf(stderr, "  item %d: text %s, tokens %zu vs %zu\n    batch: %s\n    ref  : %s\n",
                         i, b[i].text == r.text ? "same" : "DIFF", b[i].tokens.size(),
                         r.tokens.size(), b[i].text.c_str(), r.text.c_str());
            for (size_t k = 0; k < b[i].tokens.size() && k < r.tokens.size(); ++k) {
                const pk::TokenInfo &x = b[i].tokens[k], &y = r.tokens[k];
                if (x.id != y.id || x.frame != y.frame || x.span != y.span ||
                    std::fabs(x.conf - y.conf) > kBatchConfTol)
                    std::fprintf(stderr, "    tok %zu: id %d/%d frame %d/%d span %d/%d conf %g/%g\n",
                                 k, x.id, y.id, x.frame, y.frame, x.span, y.span, x.conf, y.conf);
            }
        }
    }
    std::fprintf(stderr, "  batch: %s\n", bok ? "OK" : "FAIL");
    ok = ok && bok;

    // Two threads, two different Model instances.
    pk::Transcription ta, tb;
    std::thread t1([&] { ta = model->transcribe_with_timestamps(a.samples, 16000); });
    std::thread t2([&] { tb = model2->transcribe_with_timestamps(a.samples, 16000); });
    t1.join(); t2.join();
    bool tok = ta.text == ref.text && tb.text == ref.text
            && same_tokens(ta.tokens, ref.tokens, 0.0f) && same_tokens(tb.tokens, ref.tokens, 0.0f);
    std::fprintf(stderr, "  two threads, two models: %s\n", tok ? "OK" : "FAIL");
    return ok && tok;
}

int main() {
    const char* envs[] = {"PARAKEET_TEST_GGUF_REDUX_KEEP", "PARAKEET_TEST_GGUF_ULTRA"};
    int ran = 0;
    bool ok = true;
    for (const char* e : envs) {
        const char* p = std::getenv(e);
        if (!p) continue;
        ++ran;
        ok = check_model(p) && ok;
    }
    if (!ran) { std::fprintf(stderr, "env not set; skip\n"); return 77; }
    return ok ? 0 : 1;
}
