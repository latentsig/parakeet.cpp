#pragma once
#include <string>
#include <vector>

namespace pk {

// Result of matching one embedding against the registry. An empty `name` means
// unknown; `score` is then the best cosine seen (0 for an empty registry).
struct SpeakerMatch {
    std::string name;
    float score = 0.0f;
};

// Enrolled speakers. Each speaker is the L2-normalized mean of the L2-normalized
// embeddings enrolled under its name (a centroid), so enrolling more clips
// tightens the match. Model-independent: it never sees audio. Not thread-safe.
class SpeakerRegistry {
public:
    // dim 0 means "fixed by the first enrollment".
    explicit SpeakerRegistry(int dim = 0) : dim_(dim) {}

    int dim() const { return dim_; }
    size_t size() const { return entries_.size(); }
    std::vector<std::string> names() const;   // enrollment order

    // Throws std::invalid_argument: empty name, empty or all-zero embedding,
    // or a size different from dim() when dim() != 0.
    void enroll(const std::string& name, const std::vector<float>& emb);
    bool remove(const std::string& name);

    // Best speaker by cosine. Known only if the best score >= accept and, when
    // two or more speakers are enrolled, it beats the runner-up by >= margin.
    // Throws std::invalid_argument on a size different from dim() (when dim()
    // is set). An all-zero embedding is unknown, not an error.
    SpeakerMatch identify(const std::vector<float>& emb, float accept, float margin) const;

    // Binary blob: "PKSR", u32 version 1, i32 dim, u32 n, then per speaker
    // u32 name length, name bytes, i32 count, dim x f32 sum. Little-endian.
    std::string serialize() const;
    // Throws std::runtime_error on bad magic or version, truncation, an absurd
    // count, or trailing bytes.
    static SpeakerRegistry deserialize(const std::string& blob);

private:
    struct Entry {
        std::string name;
        std::vector<float> sum;   // sum of L2-normalized enrollments
        int count = 0;
    };
    int dim_;
    std::vector<Entry> entries_;
};

}  // namespace pk
