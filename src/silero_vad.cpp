#include "silero_vad.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "backend.hpp"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml_graph.hpp"
#include "gguf.h"

namespace pk {

namespace {

void set_err(std::string* err, const std::string& m) {
    if (err) *err = m;
}

// conv1d over x [L, IC, 1] with kernel [K, IC, OC] (PyTorch weight (OC, IC, K)),
// zero padding p on both sides. Returns [OL, OC, 1]. F32 im2col, so the result
// does not lose precision (ggml_conv_1d would force an F16 im2col).
ggml_tensor* conv1d(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* x, int stride, int pad,
                    ggml_tensor* bias) {
    ggml_tensor* ic = ggml_im2col(ctx, kernel, x, stride, 0, pad, 0, 1, 0, false, GGML_TYPE_F32);  // [IC*K, OL, 1]
    ggml_tensor* y = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, ic, ic->ne[0], ic->ne[1] * ic->ne[2]),
                                  ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
    if (bias) y = ggml_add(ctx, y, ggml_reshape_2d(ctx, bias, 1, bias->ne[0]));
    return ggml_reshape_3d(ctx, y, y->ne[0], y->ne[1], 1);
}

}  // namespace

void SileroVad::Stream::reset(int sample_rate) {
    sr_ = sample_rate;
    const size_t ctx = sample_rate == 8000 ? 32 : 64;
    context_.assign(ctx, 0.0f);
    h_.assign(128, 0.0f);
    c_.assign(128, 0.0f);
}

SileroVad::~SileroVad() {
    if (buf_) ggml_backend_buffer_free(buf_);
    if (ctx_) ggml_free(ctx_);
}

const SileroVad::RateWeights* SileroVad::find(int sr) const {
    for (const auto& w : w_)
        if (w.sr == sr) return &w;
    return nullptr;
}

bool SileroVad::supports(int sr) const { return find(sr) != nullptr; }
int SileroVad::chunk_samples(int sr) const {
    const RateWeights* w = find(sr);
    return w ? w->chunk : 0;
}
double SileroVad::chunk_sec(int sr) const {
    const RateWeights* w = find(sr);
    return w ? (double)w->chunk / (double)sr : 0.0;
}

SileroVad::Stream SileroVad::new_stream(int sample_rate) const {
    Stream s;
    s.reset(sample_rate);
    return s;
}

std::unique_ptr<SileroVad> SileroVad::load(const std::string& path, std::string* err) {
    ggml_context* meta = nullptr;
    gguf_init_params gp{/*no_alloc*/ true, /*ctx*/ &meta};
    gguf_context* g = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        set_err(err, "cannot open GGUF: " + path);
        return nullptr;
    }
    struct Guard {
        gguf_context* g;
        ggml_context* m;
        ~Guard() {
            if (g) gguf_free(g);
            if (m) ggml_free(m);
        }
    } guard{g, meta};

    auto u32 = [&](const std::string& k, uint32_t& out) {
        const int64_t id = gguf_find_key(g, k.c_str());
        if (id < 0) return false;
        out = gguf_get_val_u32(g, id);
        return true;
    };
    const int64_t arch = gguf_find_key(g, "general.architecture");
    if (arch < 0 || std::string(gguf_get_val_str(g, arch)) != "silero_vad") {
        set_err(err, "not a silero_vad GGUF");
        return nullptr;
    }
    std::unique_ptr<SileroVad> m(new SileroVad());
    {
        uint32_t h = 0;
        if (!u32("silero_vad.lstm.hidden", h) || h != 128) {
            set_err(err, "unsupported LSTM size");
            return nullptr;
        }
        m->hidden_ = (int)h;
    }
    const int64_t kr = gguf_find_key(g, "silero_vad.sample_rates");
    if (kr < 0 || gguf_get_arr_type(g, kr) != GGUF_TYPE_INT32) {
        set_err(err, "missing silero_vad.sample_rates");
        return nullptr;
    }
    const int32_t* rates = (const int32_t*)gguf_get_arr_data(g, kr);
    const size_t n_rates = gguf_get_arr_n(g, kr);

    // F32 copy of every tensor in a backend buffer (F16 files are widened here).
    const int64_t nt = gguf_get_n_tensors(g);
    ggml_init_params ip{ggml_tensor_overhead() * (size_t)(nt + 4), nullptr, /*no_alloc*/ true};
    m->ctx_ = ggml_init(ip);
    if (!m->ctx_) {
        set_err(err, "ggml_init failed");
        return nullptr;
    }
    std::vector<std::pair<ggml_tensor*, int64_t>> todo;  // dst, gguf tensor id
    for (int64_t i = 0; i < nt; ++i) {
        ggml_tensor* src = ggml_get_tensor(meta, gguf_get_tensor_name(g, i));
        if (!src) continue;
        if (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) {
            set_err(err, std::string("unsupported tensor type for ") + src->name);
            return nullptr;
        }
        ggml_tensor* dst = ggml_new_tensor(m->ctx_, GGML_TYPE_F32, ggml_n_dims(src), src->ne);
        ggml_set_name(dst, src->name);
        todo.emplace_back(dst, i);
    }
    m->buf_ = ggml_backend_alloc_ctx_tensors(m->ctx_, global_backend().handle());
    if (!m->buf_) {
        set_err(err, "cannot allocate the weight buffer");
        return nullptr;
    }
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        set_err(err, "cannot reopen " + path);
        return nullptr;
    }
    const size_t base = gguf_get_data_offset(g);
    bool ok = true;
    for (auto& pr : todo) {
        ggml_tensor* src = ggml_get_tensor(meta, pr.first->name);
        const size_t n = (size_t)ggml_nelements(src);
        std::vector<uint8_t> raw(ggml_nbytes(src));
        if (std::fseek(f, (long)(base + gguf_get_tensor_offset(g, pr.second)), SEEK_SET) != 0 ||
            std::fread(raw.data(), 1, raw.size(), f) != raw.size()) {
            ok = false;
            break;
        }
        if (src->type == GGML_TYPE_F16) {
            std::vector<float> wide(n);
            ggml_fp16_to_fp32_row((const ggml_fp16_t*)raw.data(), wide.data(), (int64_t)n);
            ggml_backend_tensor_set(pr.first, wide.data(), 0, n * sizeof(float));
        } else {
            ggml_backend_tensor_set(pr.first, raw.data(), 0, raw.size());
        }
    }
    std::fclose(f);
    if (!ok) {
        set_err(err, "short read in " + path);
        return nullptr;
    }

    for (size_t r = 0; r < n_rates; ++r) {
        const int sr = rates[r];
        const std::string key = "silero_vad." + std::to_string(sr);
        const std::string pre = sr == 16000 ? "vad16k." : "vad8k.";
        RateWeights w;
        w.sr = sr;
        uint32_t chunk = 0, ctxn = 0, n_fft = 0, hop = 0, rpad = 0;
        if (!u32(key + ".chunk_samples", chunk) || !u32(key + ".context_samples", ctxn) ||
            !u32(key + ".stft.n_fft", n_fft) || !u32(key + ".stft.hop", hop) ||
            !u32(key + ".stft.right_reflect_pad", rpad)) {
            set_err(err, "missing hyper-parameters for " + std::to_string(sr) + " Hz");
            return nullptr;
        }
        w.chunk = (int)chunk, w.context = (int)ctxn, w.n_fft = (int)n_fft, w.hop = (int)hop, w.rpad = (int)rpad;
        auto get = [&](const std::string& name) -> ggml_tensor* {
            ggml_tensor* t = ggml_get_tensor(m->ctx_, (pre + name).c_str());
            if (!t) set_err(err, "missing tensor " + pre + name);
            return t;
        };
        bool all = (w.stft = get("stft.forward_basis_buffer")) != nullptr;
        for (int i = 0; i < 4 && all; ++i) {
            const std::string e = "encoder." + std::to_string(i) + ".reparam_conv.";
            all = (w.enc_w[i] = get(e + "weight")) && (w.enc_b[i] = get(e + "bias"));
        }
        all = all && (w.w_ih = get("decoder.rnn.weight_ih")) && (w.w_hh = get("decoder.rnn.weight_hh")) &&
              (w.b_ih = get("decoder.rnn.bias_ih")) && (w.b_hh = get("decoder.rnn.bias_hh")) &&
              (w.w_out = get("decoder.decoder.2.weight")) && (w.b_out = get("decoder.decoder.2.bias"));
        if (!all) return nullptr;
        // Shape checks: a wrong file must fail here and not in a ggml assert.
        const int bins = n_fft / 2 + 1;
        if (w.stft->ne[0] != n_fft || w.stft->ne[2] != 2 * bins || w.enc_w[0]->ne[1] != bins ||
            w.w_ih->ne[0] != 128 || w.w_ih->ne[1] != 512 || ggml_nelements(w.w_out) != 128 ||
            (int)chunk + (int)ctxn + (int)rpad != (int)n_fft + 3 * (int)hop) {
            set_err(err, "unexpected tensor shapes for " + std::to_string(sr) + " Hz");
            return nullptr;
        }
        m->rates_.push_back(sr);
        m->w_.push_back(w);
    }
    return m;
}

float SileroVad::process_chunk(Stream& s, const float* chunk) const {
    const RateWeights* w = find(s.sr_);
    if (!w || s.context_.size() != (size_t)w->context) return -1.0f;
    // Model input: context | chunk | right reflect padding (excluding the edge sample).
    const int L = w->context + w->chunk + w->rpad;
    std::vector<float> x((size_t)L);
    std::copy(s.context_.begin(), s.context_.end(), x.begin());
    std::copy(chunk, chunk + w->chunk, x.begin() + w->context);
    const int base = w->context + w->chunk;
    for (int i = 0; i < w->rpad; ++i) x[(size_t)base + i] = x[(size_t)base - 2 - i];

    std::vector<float> h_new, c_new, out;
    const int H = hidden_;
    const bool ok = run_graph(0, 1, [&](ggml_context* ctx) -> ggml_tensor* {
        const int64_t ne_x[3] = {L, 1, 1};
        ggml_tensor* in = graph_input_tensor(ctx, GGML_TYPE_F32, 3, ne_x, x.data(), x.size() * sizeof(float));
        const int64_t ne_h[1] = {H};
        ggml_tensor* h = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne_h, s.h_.data(), H * sizeof(float));
        ggml_tensor* c = graph_input_tensor(ctx, GGML_TYPE_F32, 1, ne_h, s.c_.data(), H * sizeof(float));

        // STFT as a strided conv with the windowed DFT basis: first half real, second imaginary.
        ggml_tensor* st = conv1d(ctx, w->stft, in, w->hop, 0, nullptr);  // [T, 2*bins, 1]
        const int64_t T = st->ne[0], bins = st->ne[1] / 2;
        ggml_tensor* re = ggml_view_2d(ctx, st, T, bins, st->nb[1], 0);
        ggml_tensor* im = ggml_view_2d(ctx, st, T, bins, st->nb[1], (size_t)bins * st->nb[1]);
        ggml_tensor* mag = ggml_sqrt(ctx, ggml_add(ctx, ggml_mul(ctx, re, re), ggml_mul(ctx, im, im)));
        ggml_tensor* cur = ggml_reshape_3d(ctx, mag, T, bins, 1);

        static const int kStride[4] = {1, 2, 2, 1};
        for (int i = 0; i < 4; ++i) cur = ggml_relu(ctx, conv1d(ctx, w->enc_w[i], cur, kStride[i], 1, w->enc_b[i]));
        // Time length is 1 here: [1, 128, 1] is the LSTM input.
        ggml_tensor* xt = ggml_reshape_2d(ctx, cur, H, 1);
        ggml_tensor* g = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w->w_ih, xt), ggml_reshape_2d(ctx, w->b_ih, 512, 1)),
                                  ggml_add(ctx, ggml_mul_mat(ctx, w->w_hh, ggml_reshape_2d(ctx, h, H, 1)),
                                           ggml_reshape_2d(ctx, w->b_hh, 512, 1)));
        auto gate = [&](int k) { return ggml_view_1d(ctx, g, H, (size_t)k * H * sizeof(float)); };  // i, f, g, o
        ggml_tensor* c2 = ggml_add(ctx, ggml_mul(ctx, ggml_sigmoid(ctx, gate(1)), c),
                                   ggml_mul(ctx, ggml_sigmoid(ctx, gate(0)), ggml_tanh(ctx, gate(2))));
        ggml_tensor* h2 = ggml_mul(ctx, ggml_sigmoid(ctx, gate(3)), ggml_tanh(ctx, c2));
        capture_graph_output(h2, &h_new);
        capture_graph_output(c2, &c_new);
        ggml_tensor* z = ggml_add(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w->w_out, H, 1), ggml_reshape_2d(ctx, ggml_relu(ctx, h2), H, 1)),
                                  ggml_reshape_2d(ctx, w->b_out, 1, 1));
        return ggml_sigmoid(ctx, z);
    }, out);
    if (!ok || out.size() != 1 || h_new.size() != (size_t)H || c_new.size() != (size_t)H) return -1.0f;
    s.h_ = std::move(h_new);
    s.c_ = std::move(c_new);
    std::copy(x.begin() + w->chunk, x.begin() + w->chunk + w->context, s.context_.begin());
    return out[0];
}

std::vector<float> SileroVad::probabilities(const float* pcm, size_t n, int sample_rate) const {
    std::vector<float> p;
    const RateWeights* w = find(sample_rate);
    if (!w) return p;
    Stream s = new_stream(sample_rate);
    std::vector<float> tmp((size_t)w->chunk);
    for (size_t i = 0; i < n; i += (size_t)w->chunk) {
        const size_t k = std::min((size_t)w->chunk, n - i);
        std::fill(tmp.begin(), tmp.end(), 0.0f);
        std::copy(pcm + i, pcm + i + k, tmp.begin());
        const float v = process_chunk(s, tmp.data());
        if (v < 0.0f) return {};
        p.push_back(v);
    }
    return p;
}

}  // namespace pk
