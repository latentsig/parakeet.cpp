// Unit test for pk::SpeakerRegistry. No model or audio needed.
#include "speaker_registry.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pk;

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__);  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static std::vector<float> unit(int dim, int hot) {
    std::vector<float> v((size_t)dim, 0.0f);
    v[(size_t)hot] = 1.0f;
    return v;
}

static void test_enroll_and_identify() {
    SpeakerRegistry r;
    r.enroll("alice", unit(4, 0));
    r.enroll("bob", unit(4, 1));
    CHECK(r.dim() == 4);
    CHECK(r.size() == 2);
    SpeakerMatch m = r.identify(unit(4, 0), 0.5f, 0.05f);
    CHECK(m.name == "alice");
    CHECK(std::fabs(m.score - 1.0f) < 1e-5f);
    m = r.identify(unit(4, 1), 0.5f, 0.05f);
    CHECK(m.name == "bob");
}

static void test_centroid_averages_enrollments() {
    SpeakerRegistry r;
    r.enroll("alice", {1.0f, 0.0f});
    r.enroll("alice", {0.0f, 1.0f});   // centroid points at (1,1)/sqrt2
    CHECK(r.size() == 1);
    const SpeakerMatch m = r.identify({1.0f, 1.0f}, 0.9f, 0.0f);
    CHECK(m.name == "alice");
    CHECK(m.score > 0.999f);
}

static void test_unknown_below_threshold() {
    SpeakerRegistry r;
    r.enroll("alice", unit(4, 0));
    const SpeakerMatch m = r.identify(unit(4, 2), 0.5f, 0.05f);   // orthogonal: cosine 0
    CHECK(m.name.empty());
    CHECK(std::fabs(m.score) < 1e-5f);
}

static void test_margin() {
    SpeakerRegistry r;
    r.enroll("alice", {1.0f, 0.0f});
    r.enroll("bob", {0.0f, 1.0f});
    // Equidistant probe: both score 0.707, so the margin is 0 and it must be unknown.
    const SpeakerMatch m = r.identify({1.0f, 1.0f}, 0.5f, 0.05f);
    CHECK(m.name.empty());
    CHECK(m.score > 0.7f);
    // A single enrolled speaker has no runner-up, so the margin does not apply.
    SpeakerRegistry one;
    one.enroll("alice", {1.0f, 0.0f});
    CHECK(one.identify({1.0f, 1.0f}, 0.5f, 0.5f).name == "alice");
}

static void test_dim_mismatch() {
    SpeakerRegistry r;
    r.enroll("alice", unit(4, 0));
    bool threw = false;
    try { r.enroll("bob", unit(3, 0)); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { r.identify(unit(3, 0), 0.5f, 0.05f); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_bad_enroll() {
    SpeakerRegistry r;
    auto throws = [&](const std::string& n, const std::vector<float>& e) {
        try { r.enroll(n, e); } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    CHECK(throws("", unit(4, 0)));
    CHECK(throws("a", {}));
    CHECK(throws("a", {0.0f, 0.0f}));
    CHECK(r.size() == 0);
    // An all-zero probe is unknown, not an error.
    r.enroll("alice", unit(2, 0));
    CHECK(r.identify({0.0f, 0.0f}, 0.5f, 0.05f).name.empty());
}

static void test_remove_and_names() {
    SpeakerRegistry r;
    r.enroll("bob", unit(2, 1));
    r.enroll("alice", unit(2, 0));
    const auto n = r.names();
    CHECK(n.size() == 2 && n[0] == "bob" && n[1] == "alice");   // enrollment order
    CHECK(r.remove("bob"));
    CHECK(!r.remove("bob"));
    CHECK(r.size() == 1);
}

static void test_serialize_roundtrip() {
    SpeakerRegistry r;
    r.enroll("alice", {1.0f, 0.0f, 0.0f});
    r.enroll("alice", {0.9f, 0.1f, 0.0f});
    r.enroll("bob", {0.0f, 1.0f, 0.0f});
    const SpeakerRegistry back = SpeakerRegistry::deserialize(r.serialize());
    CHECK(back.dim() == 3);
    CHECK(back.size() == 2);
    const auto a = r.identify({0.95f, 0.05f, 0.0f}, 0.5f, 0.05f);
    const auto b = back.identify({0.95f, 0.05f, 0.0f}, 0.5f, 0.05f);
    CHECK(a.name == b.name && std::fabs(a.score - b.score) < 1e-6f);
    // Enrolling more into the loaded registry keeps averaging (count survived).
    SpeakerRegistry loaded = SpeakerRegistry::deserialize(r.serialize());
    loaded.enroll("alice", {0.0f, 0.0f, 1.0f});
    CHECK(loaded.size() == 2);
}

static void test_deserialize_corrupt() {
    SpeakerRegistry r;
    r.enroll("alice", {1.0f, 0.0f});
    const std::string good = r.serialize();
    auto throws = [](const std::string& s) {
        try { SpeakerRegistry::deserialize(s); } catch (const std::runtime_error&) { return true; }
        return false;
    };
    CHECK(throws(""));
    CHECK(throws("not a registry"));
    CHECK(throws(good.substr(0, good.size() - 1)));   // truncated
    CHECK(throws(good + "x"));                        // trailing bytes
    std::string bad_magic = good;
    bad_magic[0] = 'X';
    CHECK(throws(bad_magic));
    std::string huge = good;                          // absurd speaker count
    huge[12] = (char)0xff; huge[13] = (char)0xff; huge[14] = (char)0xff; huge[15] = (char)0x7f;
    CHECK(throws(huge));
}

int main() {
    test_enroll_and_identify();
    test_centroid_averages_enrollments();
    test_unknown_below_threshold();
    test_margin();
    test_dim_mismatch();
    test_bad_enroll();
    test_remove_and_names();
    test_serialize_roundtrip();
    test_deserialize_corrupt();
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_speaker_registry: PASS\n");
    return 0;
}
