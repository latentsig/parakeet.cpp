#include "speaker_encoder.hpp"

#include "gguf.h"

#ifdef PARAKEET_WITH_VOICEDETECT
#include "voicedetect_capi.h"
#endif

namespace pk {

bool gguf_is_voicedetect(const std::string& path) {
    gguf_init_params p{/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) return false;
    const int64_t id = gguf_find_key(g, "general.architecture");
    const bool vd = id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING &&
                    std::string(gguf_get_val_str(g, id)) == "voicedetect";
    gguf_free(g);
    return vd;
}

#ifdef PARAKEET_WITH_VOICEDETECT

bool SpeakerEncoder::available() { return true; }

std::unique_ptr<SpeakerEncoder> SpeakerEncoder::load(const std::string& path) {
    voicedetect_ctx* c = voicedetect_capi_load(path.c_str());
    if (!c) return nullptr;
    const int dim = voicedetect_capi_embedding_dim(c);
    if (dim <= 0) {   // an analyze-only model has no speaker embedding
        voicedetect_capi_free(c);
        return nullptr;
    }
    std::unique_ptr<SpeakerEncoder> e(new SpeakerEncoder());
    e->ctx_ = c;
    e->dim_ = dim;
    return e;
}

SpeakerEncoder::~SpeakerEncoder() { voicedetect_capi_free(static_cast<voicedetect_ctx*>(ctx_)); }

bool SpeakerEncoder::embed(const float* pcm, int n, std::vector<float>& emb) {
    last_error_.clear();   // describes the latest call only
    if (!pcm || n <= 0) {
        last_error_ = "no audio to embed";
        return false;
    }
    auto* c = static_cast<voicedetect_ctx*>(ctx_);
    float* v = nullptr;
    int d = 0;
    if (voicedetect_capi_embed_pcm(c, pcm, n, 16000, &v, &d) != 0 || !v || d != dim_) {
        const char* m = voicedetect_capi_last_error(c);
        last_error_ = (m && *m) ? m : "speaker embedding failed";
        voicedetect_capi_free_vec(v);
        return false;
    }
    emb.assign(v, v + d);
    voicedetect_capi_free_vec(v);
    return true;
}

#else  // PARAKEET_WITH_VOICEDETECT

bool SpeakerEncoder::available() { return false; }
std::unique_ptr<SpeakerEncoder> SpeakerEncoder::load(const std::string&) { return nullptr; }
SpeakerEncoder::~SpeakerEncoder() = default;
bool SpeakerEncoder::embed(const float*, int, std::vector<float>&) {
    last_error_ = "built without speaker identification (PARAKEET_WITH_VOICEDETECT=OFF)";
    return false;
}

#endif

SpeakerEmbed SpeakerEncoder::embedder() {
    return [this](const float* pcm, int n, std::vector<float>& emb) { return embed(pcm, n, emb); };
}

}  // namespace pk
