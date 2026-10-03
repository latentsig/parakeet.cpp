#include "vad_json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "backend.hpp"
#include "ggml_graph.hpp"
#include "model.hpp"
#include "silero_vad.hpp"

#include "gguf.h"

namespace pk {

namespace {

struct Cursor {
    const char* p;
    void ws() { while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') ++p; }
};

bool parse_string(Cursor& c, std::string& out) {
    if (*c.p != '"') return false;
    ++c.p;
    out.clear();
    while (*c.p && *c.p != '"') {
        if (*c.p == '\\') return false;  // no escapes are needed for any key or value
        out += *c.p++;
    }
    if (*c.p != '"') return false;
    ++c.p;
    return true;
}

bool parse_number(Cursor& c, double& v) {
    char* end = nullptr;
    v = std::strtod(c.p, &end);
    if (end == c.p || !std::isfinite(v)) return false;
    c.p = end;
    return true;
}

bool positive_seconds(double v) { return std::isfinite(v) && v > 0.0 && v <= 1e6; }

}  // namespace

bool parse_vad_options(const char* json, VadRequest& req, std::string& err, VadKind kind) {
    req = VadRequest();
    req.kind = kind;
    req.opts = default_segmenter_opts(kind);
    if (!json) return true;
    Cursor c{json};
    c.ws();
    if (*c.p == '\0') return true;
    if (*c.p != '{') { err = "invalid VAD options: expected a JSON object"; return false; }
    ++c.p;
    c.ws();
    if (*c.p == '}') { ++c.p; c.ws(); if (*c.p) { err = "invalid VAD options: trailing text"; return false; } return true; }
    for (;;) {
        c.ws();
        std::string key;
        if (!parse_string(c, key)) { err = "invalid VAD options: expected a key"; return false; }
        c.ws();
        if (*c.p != ':') { err = "invalid VAD options: expected ':' after \"" + key + "\""; return false; }
        ++c.p;
        c.ws();
        if (key == "mode") {
            std::string v;
            if (!parse_string(c, v)) { err = "invalid VAD option mode: expected a string"; return false; }
            if (v == "speech") req.mode = VadRequest::Mode::kSpeech;
            else if (v == "segments") req.mode = VadRequest::Mode::kSegments;
            else { err = "invalid VAD option mode: use \"speech\" or \"segments\""; return false; }
        } else if (key == "probabilities") {
            if (std::strncmp(c.p, "true", 4) == 0) { req.probabilities = true; c.p += 4; }
            else if (std::strncmp(c.p, "false", 5) == 0) { req.probabilities = false; c.p += 5; }
            else { err = "invalid VAD option probabilities: expected true or false"; return false; }
        } else if (key == "threshold" || key == "min_pause" || key == "min_speech" ||
                   key == "max_segment" || key == "speech_pad") {
            double v = 0.0;
            if (!parse_number(c, v)) { err = "invalid VAD option " + key + ": expected a number"; return false; }
            if (key == "threshold") {
                if (!(v > 0.0 && v <= 1.0)) { err = "invalid VAD option threshold: must be in (0, 1]"; return false; }
                req.opts.threshold = (float)v;
            } else if (key == "speech_pad") {
                if (!(std::isfinite(v) && v >= 0.0 && v <= 1e6)) { err = "invalid VAD option speech_pad: must be a number of seconds >= 0"; return false; }
                req.opts.pad_sec = v;
            } else {
                if (!positive_seconds(v)) { err = "invalid VAD option " + key + ": must be a number of seconds > 0"; return false; }
                if (key == "min_pause") req.opts.min_pause_sec = v;
                else if (key == "min_speech") req.opts.min_speech_sec = v;
                else req.opts.max_seg_sec = v;
            }
        } else {
            err = "unknown VAD option: " + key;
            return false;
        }
        c.ws();
        if (*c.p == ',') { ++c.p; continue; }
        if (*c.p == '}') { ++c.p; break; }
        err = "invalid VAD options: expected ',' or '}'";
        return false;
    }
    c.ws();
    if (*c.p) { err = "invalid VAD options: trailing text"; return false; }
    return true;
}

namespace {

std::string format_vad_json(const VadRequest& req, double total, double frame_sec,
                            const std::vector<float>& p, const std::vector<VadSegment>& segs) {
    char buf[64];
    std::string j = "{\"mode\":\"";
    j += req.mode == VadRequest::Mode::kSpeech ? "speech" : "segments";
    std::snprintf(buf, sizeof(buf), "\",\"duration\":%.3f,\"frame_sec\":%.3f,\"backend\":\"",
                  total, frame_sec);
    j += buf;
    j += pk::global_backend().device_name();
    j += "\",\"segments\":[";
    for (size_t i = 0; i < segs.size(); ++i) {
        std::snprintf(buf, sizeof(buf), "%s{\"start\":%.3f,\"end\":%.3f}", i ? "," : "",
                      segs[i].start, segs[i].end);
        j += buf;
    }
    j += ']';
    if (req.probabilities) {
        j += ",\"probabilities\":[";
        for (size_t i = 0; i < p.size(); ++i) {
            std::snprintf(buf, sizeof(buf), "%s%.4f", i ? "," : "", (double)p[i]);
            j += buf;
        }
        j += ']';
    }
    j += '}';
    return j;
}

}  // namespace

std::string vad_to_json(const Model& m, const std::vector<float>& pcm16k, const VadRequest& req) {
    const ParakeetConfig& cfg = m.config();
    if (!cfg.vad.present) throw std::runtime_error("model has no VAD head");
    SegmenterOpts o = req.opts;
    o.frame_sec = cfg.vad.frame_sec;
    if (req.mode == VadRequest::Mode::kSegments && !(o.max_seg_sec > 2.0 * o.frame_sec))
        throw std::runtime_error("invalid VAD option max_segment: too small");
    const double total = (double)pcm16k.size() / 16000.0;
    std::vector<float> p;
    if (!pcm16k.empty()) p = m.vad_probabilities(pcm16k);
    std::vector<VadSegment> segs;
    if (!pcm16k.empty())
        segs = req.mode == VadRequest::Mode::kSpeech ? speech_regions(p, total, o)
                                                      : segment_by_vad(p, total, o);
    return format_vad_json(req, total, (double)o.frame_sec, p, segs);
}

std::string silero_vad_to_json(const SileroVad& m, const std::vector<float>& pcm, int sample_rate,
                               const VadRequest& req) {
    if (!m.supports(sample_rate))
        throw std::runtime_error("Silero VAD needs 16000 or 8000 Hz audio");
    SegmenterOpts o = req.opts;
    o.frame_sec = kSileroFrameSec;
    if (req.mode == VadRequest::Mode::kSegments && !(o.max_seg_sec > 2.0 * o.frame_sec))
        throw std::runtime_error("invalid VAD option max_segment: too small");
    const double total = (double)pcm.size() / (double)sample_rate;
    std::vector<float> p;
    if (!pcm.empty()) {
        p = m.probabilities(pcm.data(), pcm.size(), sample_rate);
        if (p.empty()) throw std::runtime_error("Silero VAD failed");
    }
    std::vector<VadSegment> segs;
    if (!pcm.empty())
        segs = req.mode == VadRequest::Mode::kSpeech ? speech_regions(p, total, o)
                                                      : segment_by_vad(p, total, o);
    return format_vad_json(req, total, o.frame_sec, p, segs);
}

bool gguf_is_silero(const std::string& path) {
    gguf_init_params ip{/*no_alloc=*/true, /*ctx=*/nullptr};
    gguf_context* g = gguf_init_from_file(path.c_str(), ip);
    if (!g) return false;
    const int64_t id = gguf_find_key(g, "general.architecture");
    const bool ok = id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING &&
                    std::string(gguf_get_val_str(g, id)) == "silero_vad";
    gguf_free(g);
    return ok;
}

}  // namespace pk
