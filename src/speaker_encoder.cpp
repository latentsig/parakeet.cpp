#include "speaker_encoder.hpp"

#include "bundle.hpp"
#include "bundle_extract.hpp"
#include "speaker_model_identity.hpp"
#include "gguf.h"

#include <stdexcept>

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

std::string speaker_encoder_family(const std::string& path, int dim_fallback) {
    gguf_init_params p{/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) return "";
    auto str = [&](const char* key) {
        const int64_t id = gguf_find_key(g, key);
        return (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING)
                   ? std::string(gguf_get_val_str(g, id)) : std::string();
    };
    std::string out;
    if (str("general.architecture") == "voicedetect") {
        long long dim = dim_fallback;
        const int64_t id = gguf_find_key(g, "voicedetect.embedding_dim");
        if (id >= 0) {
            const gguf_type t = gguf_get_kv_type(g, id);
            if (t == GGUF_TYPE_UINT32) dim = gguf_get_val_u32(g, id);
            else if (t == GGUF_TYPE_INT32) dim = gguf_get_val_i32(g, id);
        }
        out = "voicedetect:" + str("voicedetect.arch") + ":" + str("general.name") + ":" +
              (dim > 0 ? std::to_string(dim) : std::string());
    }
    gguf_free(g);
    return out;
}

#ifdef PARAKEET_WITH_VOICEDETECT

bool SpeakerEncoder::available() { return true; }

// The weights identity of a standalone file is the sha256 of its bytes, taken on both
// sides of the load: a file replaced while loading is refused. Model files must stay
// immutable in service. The identity of a bundle component is the sha256 of the
// single-model file it was built from, recorded in the bundle header, so an encoder
// enrolled from the standalone file matches the same encoder inside a bundle.
std::unique_ptr<SpeakerEncoder> SpeakerEncoder::load(const std::string& path, const std::string& component,
                                                     std::string* err) {
    std::string weights;
    std::unique_ptr<ComponentFile> cf;
    std::string load_path = path;
    try {
        if (component.empty()) {
            weights = speaker_model_identity(path);
        } else {
            BundleInfo info;
            std::string e;
            if (!read_bundle_info(path, info, &e)) { if (err) *err = e; return nullptr; }
            const BundleComponent* c = info.find(component);
            if (!c) { if (err) *err = "the bundle has no component \"" + component + "\""; return nullptr; }
            const std::string& h = c->source_sha256;
            bool hex = h.size() == 64;
            for (char ch : h) hex = hex && ((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'));
            if (!hex) {
                if (err) *err = "voice component \"" + component + "\" has no valid source_sha256 in the bundle "
                                "header, which is its speaker model identity";
                return nullptr;
            }
            weights = "sha256:" + h;
            cf = ComponentFile::create(path, component, &e);
            if (!cf) { if (err) *err = e; return nullptr; }
            load_path = cf->path();   // the file is removed when cf goes out of scope
        }
        std::unique_ptr<SpeakerEncoder> enc = load_unchecked(load_path, weights, err);
        if (enc && component.empty() && speaker_model_identity(path) != weights) {
            if (err) *err = "the speaker model changed during load";
            return nullptr;
        }
        return enc;
    } catch (...) {
        if (err && err->empty()) *err = "cannot read the speaker model " + (component.empty() ? path : component);
        return nullptr;
    }
}

std::unique_ptr<SpeakerEncoder> SpeakerEncoder::load_unchecked(const std::string& path,
                                                                const std::string& weights,
                                                                std::string* err) {
    voicedetect_ctx* c = voicedetect_capi_load(path.c_str());
    if (!c) {
        if (err && err->empty()) *err = "cannot load the speaker model";
        return nullptr;
    }
    const int dim = voicedetect_capi_embedding_dim(c);
    if (dim <= 0) {   // an analyze-only model has no speaker embedding
        voicedetect_capi_free(c);
        if (err) *err = "the speaker model has no speaker embedding (an analysis model?)";
        return nullptr;
    }
    std::unique_ptr<SpeakerEncoder> e(new SpeakerEncoder());
    e->ctx_ = c;
    e->dim_ = dim;
    e->fp_.family = speaker_encoder_family(path, dim);
    e->fp_.weights = weights;
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
std::unique_ptr<SpeakerEncoder> SpeakerEncoder::load(const std::string&, const std::string&, std::string* err) {
    if (err) *err = "this build has no speaker identification (PARAKEET_WITH_VOICEDETECT=OFF)";
    return nullptr;
}
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
