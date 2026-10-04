#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace pk {

// Scores one PCM window (16 kHz mono): fills `probs` with one score per class
// in class-index order and returns true, or returns false on failure.
using SoundScorer = std::function<bool(const float* pcm, int n, std::vector<float>& probs)>;

// A loaded ced.cpp model (CED AudioSet tagger). The only parakeet code that
// talks to ced.cpp, and only through ced_capi.h. Not thread-safe: one stream
// at a time per tagger, like the other contexts.
class CedTagger {
public:
    // False when parakeet was built with PARAKEET_WITH_CED=OFF.
    static bool available();
    // nullptr on failure (or when unavailable). With a non-empty `component`,
    // `gguf_path` is a bundle GGUF (docs/bundle.md) and the model is its component
    // of kind "ced". ced.cpp opens models by path only, so the component is first
    // written as a standalone GGUF (an in-memory file on Linux, else a temporary
    // file removed after the load; see bundle_extract.hpp). `err`, when given,
    // receives the reason for a failure.
    static std::unique_ptr<CedTagger> load(const std::string& gguf_path, const std::string& component = "",
                                           std::string* err = nullptr);
    ~CedTagger();
    CedTagger(const CedTagger&) = delete;
    CedTagger& operator=(const CedTagger&) = delete;

    int n_classes() const;
    const char* label(int index) const;
    SoundScorer scorer();
    const std::string& last_error() const { return last_error_; }

private:
    CedTagger() = default;
    void* ctx_ = nullptr;  // ced_ctx*
    std::string last_error_;
};

// True when the GGUF's general.architecture is "ced". Reads only the header.
bool gguf_is_ced(const std::string& gguf_path);

} // namespace pk
