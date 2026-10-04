// Unit test for pk::SpeakerRegistry. No model or audio needed.
#include "speaker_registry.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
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
    // Failed enroll on non-empty registry leaves state unchanged.
    const int orig_dim = r.dim();
    const size_t orig_size = r.size();
    const SpeakerMatch orig_match = r.identify(unit(2, 0), 0.5f, 0.05f);
    CHECK(throws("alice", unit(3, 0)));   // dim mismatch
    CHECK(throws("bob", {0.0f, 0.0f}));   // zero vector
    CHECK(r.dim() == orig_dim && r.size() == orig_size);
    const SpeakerMatch new_match = r.identify(unit(2, 0), 0.5f, 0.05f);
    CHECK(new_match.name == orig_match.name && std::fabs(new_match.score - orig_match.score) < 1e-5f);
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
    // names() still lists remaining speaker in order after remove.
    const auto remaining = r.names();
    CHECK(remaining.size() == 1 && remaining[0] == "alice");
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
    // Corrupt: dim 0 with n > 0 (would cause out-of-bounds write on later enroll).
    // Build manually: "PKSR" (4 bytes) + version 1 (4 bytes, little-endian) +
    // dim 0 (4 bytes) + n 1 (4 bytes) + name length 5 (4 bytes) + "alice" (5 bytes) +
    // count 1 (4 bytes) + 0 floats for sum (since dim is 0).
    std::string corrupt_dim_zero;
    corrupt_dim_zero += 'P'; corrupt_dim_zero += 'K'; corrupt_dim_zero += 'S'; corrupt_dim_zero += 'R';
    // version 1 in little-endian
    corrupt_dim_zero += (char)0x01; corrupt_dim_zero += (char)0x00; corrupt_dim_zero += (char)0x00; corrupt_dim_zero += (char)0x00;
    // dim 0 in little-endian
    corrupt_dim_zero += (char)0x00; corrupt_dim_zero += (char)0x00; corrupt_dim_zero += (char)0x00; corrupt_dim_zero += (char)0x00;
    // n 1 in little-endian
    corrupt_dim_zero += (char)0x01; corrupt_dim_zero += (char)0x00; corrupt_dim_zero += (char)0x00; corrupt_dim_zero += (char)0x00;
    // name length 5 in little-endian
    corrupt_dim_zero += (char)0x05; corrupt_dim_zero += (char)0x00; corrupt_dim_zero += (char)0x00; corrupt_dim_zero += (char)0x00;
    // name "alice"
    corrupt_dim_zero += "alice";
    // count 1 in little-endian
    corrupt_dim_zero += (char)0x01; corrupt_dim_zero += (char)0x00; corrupt_dim_zero += (char)0x00; corrupt_dim_zero += (char)0x00;
    CHECK(throws(corrupt_dim_zero));
    // Empty registry (dim 0, n 0) round-trips and accepts normal enroll.
    SpeakerRegistry empty;
    const std::string empty_blob = empty.serialize();
    SpeakerRegistry loaded_empty = SpeakerRegistry::deserialize(empty_blob);
    CHECK(loaded_empty.dim() == 0 && loaded_empty.size() == 0);
    loaded_empty.enroll("charlie", {1.0f, 0.0f});
    CHECK(loaded_empty.size() == 1 && loaded_empty.dim() == 2);
}

// A NaN or Inf embedding is refused like an all-zero one, and a NaN probe is unknown.
static void test_non_finite() {
    SpeakerRegistry r;
    r.enroll("alice", unit(2, 0));
    const SpeakerMatch before = r.identify(unit(2, 0), 0.5f, 0.05f);
    auto throws = [&](const std::string& n, const std::vector<float>& e, std::string* msg) {
        try { r.enroll(n, e); } catch (const std::invalid_argument& ex) { *msg = ex.what(); return true; }
        return false;
    };
    std::string msg;
    CHECK(throws("bob", {std::nanf(""), 1.0f}, &msg));
    CHECK(msg.find("not finite") != std::string::npos);
    CHECK(throws("bob", {INFINITY, 0.0f}, &msg));
    CHECK(throws("alice", {-INFINITY, 1.0f}, &msg));   // also for a name already enrolled
    CHECK(r.size() == 1 && r.dim() == 2);
    const SpeakerMatch after = r.identify(unit(2, 0), 0.5f, 0.05f);
    CHECK(after.name == before.name && std::fabs(after.score - before.score) < 1e-6f);
    const SpeakerMatch nan_probe = r.identify({std::nanf(""), 1.0f}, 0.5f, 0.05f);
    CHECK(nan_probe.name.empty());
    CHECK(r.identify({INFINITY, 0.0f}, 0.5f, 0.05f).name.empty());
    SpeakerRegistry empty;   // a failed first enroll does not fix the dimension
    try { empty.enroll("x", {std::nanf(""), 0.0f, 0.0f}); } catch (const std::invalid_argument&) {}
    CHECK(empty.size() == 0 && empty.dim() == 0);
}


static bool throws_deser(const std::string& blob) {
    try { SpeakerRegistry::deserialize(blob); } catch (const std::runtime_error&) { return true; }
    return false;
}

// ---- encoder fingerprint ---------------------------------------------------

static const EncoderFingerprint kFpA{"voicedetect:wespeaker_resnet34:wespeaker:256", "sha256:aaaa"};
static const EncoderFingerprint kFpA16{"voicedetect:wespeaker_resnet34:wespeaker:256", "sha256:bbbb"};
static const EncoderFingerprint kFpB{"voicedetect:ecapa_tdnn:ecapa:192", "sha256:cccc"};
static const EncoderFingerprint kFpC{"voicedetect:campplus:campplus:192", "sha256:dddd"};

static void put_u32(std::string& s, uint32_t v) { s.append(reinterpret_cast<const char*>(&v), 4); }

static void test_fp_enroll_stamps_and_roundtrips() {
    SpeakerRegistry r;
    CHECK(r.fingerprint().empty());
    CHECK(r.enroll("alice", unit(4, 0), kFpA).status == FingerprintStatus::Match);
    CHECK(r.fingerprint().family == kFpA.family && r.fingerprint().weights == kFpA.weights);
    const SpeakerRegistry back = SpeakerRegistry::deserialize(r.serialize());
    CHECK(back.fingerprint().family == kFpA.family && back.fingerprint().weights == kFpA.weights);
    CHECK(back.size() == 1 && back.dim() == 4);
    CHECK(back.identify(unit(4, 0), 0.5f, 0.05f).name == "alice");
    // A fingerprint with an empty weights hash is kept as given.
    SpeakerRegistry fam;
    fam.enroll("a", unit(4, 0), EncoderFingerprint{"fam", ""});
    CHECK(SpeakerRegistry::deserialize(fam.serialize()).fingerprint().family == "fam");
}

static void test_fp_enroll_rules() {
    SpeakerRegistry r;
    r.enroll("alice", unit(4, 0), kFpA);
    // Same family, other weights: allowed, reported, registry keeps its own hash.
    FingerprintVerdict v = r.enroll("bob", unit(4, 1), kFpA16);
    CHECK(v.status == FingerprintStatus::WeightsDiffer && !v.message.empty());
    CHECK(r.fingerprint().weights == kFpA.weights && r.size() == 2);
    // Other family: refused, nothing added.
    std::string msg;
    try { r.enroll("cy", unit(4, 2), kFpC); } catch (const std::invalid_argument& e) { msg = e.what(); }
    CHECK(msg.find(kFpA.family) != std::string::npos && msg.find(kFpC.family) != std::string::npos);
    CHECK(r.size() == 2);
    // No fingerprint into a fingerprinted registry: refused.
    msg.clear();
    try { r.enroll("dee", unit(4, 3)); } catch (const std::invalid_argument& e) { msg = e.what(); }
    CHECK(!msg.empty() && r.size() == 2);
    // A fingerprint into a non-empty unfingerprinted registry: refused, never stamped silently.
    SpeakerRegistry old;
    old.enroll("alice", unit(4, 0));
    msg.clear();
    try { old.enroll("bob", unit(4, 1), kFpA); } catch (const std::invalid_argument& e) { msg = e.what(); }
    CHECK(!msg.empty() && old.size() == 1 && old.fingerprint().empty());
    // An empty fingerprint behaves as before.
    SpeakerRegistry plain;
    plain.enroll("alice", unit(4, 0), EncoderFingerprint{});
    CHECK(plain.fingerprint().empty());
}

static void test_fp_check() {
    // Match.
    CHECK(check_fingerprint(kFpA, kFpA, false).status == FingerprintStatus::Match);
    CHECK(check_fingerprint(kFpA, kFpA, true).status == FingerprintStatus::Match);
    // Same family, other weights: a warning, even in strict mode.
    FingerprintVerdict w = check_fingerprint(kFpA, kFpA16, true);
    CHECK(w.status == FingerprintStatus::WeightsDiffer && !w.is_error() && !w.message.empty());
    // Same dimension, other family: a hard error that names both families.
    FingerprintVerdict f = check_fingerprint(kFpB, kFpC, false);
    CHECK(f.status == FingerprintStatus::FamilyDiffer && f.is_error());
    CHECK(f.message.find(kFpB.family) != std::string::npos);
    CHECK(f.message.find(kFpC.family) != std::string::npos);
    // No fingerprint: soft warning, or an error in strict mode.
    FingerprintVerdict u = check_fingerprint(EncoderFingerprint{}, kFpA, false);
    CHECK(u.status == FingerprintStatus::Unfingerprinted && !u.is_error() && !u.message.empty());
    FingerprintVerdict s = check_fingerprint(EncoderFingerprint{}, kFpA, true);
    CHECK(s.status == FingerprintStatus::Required && s.is_error() && !s.message.empty());
    // Family only in the registry: weights are not compared.
    CHECK(check_fingerprint(EncoderFingerprint{kFpA.family, ""}, kFpA16, true).status == FingerprintStatus::Match);
    // An encoder that cannot name its family is not checked.
    CHECK(check_fingerprint(kFpA, EncoderFingerprint{}, true).status == FingerprintStatus::Match);
}

static void test_fp_registry_check() {
    SpeakerRegistry r;
    r.enroll("a", unit(192, 0), kFpB);
    FingerprintVerdict v = check_registry_for_encoder(r, 256, kFpA, false);
    CHECK(v.status == FingerprintStatus::DimDiffer && v.is_error());
    CHECK(v.message.find("192") != std::string::npos && v.message.find("256") != std::string::npos);
    CHECK(check_registry_for_encoder(r, 192, kFpC, false).status == FingerprintStatus::FamilyDiffer);
    CHECK(check_registry_for_encoder(r, 192, kFpB, true).status == FingerprintStatus::Match);
    SpeakerRegistry empty;   // nothing enrolled: nothing to mix up, even in strict mode
    CHECK(check_registry_for_encoder(empty, 192, kFpB, true).status == FingerprintStatus::Match);
}

static void test_fp_v1_compat() {
    // v1 bytes exactly as the previous release wrote them.
    std::string v1("PKSR", 4);
    put_u32(v1, 1);
    put_u32(v1, 2);   // dim
    put_u32(v1, 1);   // n
    put_u32(v1, 5);
    v1 += "alice";
    put_u32(v1, 1);   // count
    const float sum[2] = {1.0f, 0.0f};
    v1.append(reinterpret_cast<const char*>(sum), sizeof(sum));
    SpeakerRegistry r = SpeakerRegistry::deserialize(v1);
    CHECK(r.size() == 1 && r.dim() == 2 && r.fingerprint().empty());
    CHECK(r.identify(unit(2, 0), 0.5f, 0.05f).name == "alice");
    // An unfingerprinted registry is still written as v1, byte for byte.
    CHECK(r.serialize() == v1);
    // A fingerprinted one is v2, and a reader that only knows v1 refuses it.
    r.set_fingerprint(kFpA);
    const std::string v2 = r.serialize();
    CHECK(v2.size() > v1.size());
    uint32_t ver = 0;
    std::memcpy(&ver, v2.data() + 4, 4);
    CHECK(ver == 2);
    std::string patched = v2;
    const uint32_t three = 3;
    std::memcpy(&patched[4], &three, 4);
    CHECK(throws_deser(patched));   // unknown version
    std::string as_v1 = v2;
    const uint32_t one = 1;
    std::memcpy(&as_v1[4], &one, 4);
    CHECK(throws_deser(as_v1));     // a v1 parser meets the extra bytes: refused, no crash
    // Truncation inside the fingerprint block.
    CHECK(throws_deser(v2.substr(0, 4 + 4 + 4 + 4 + 2)));
    CHECK(throws_deser(v2.substr(0, 4 + 4 + 4 + 4 + 4 + 3)));
    // An absurd fingerprint length is refused.
    std::string big = v2;
    const uint32_t huge = 0x7fffffff;
    std::memcpy(&big[16], &huge, 4);
    CHECK(throws_deser(big));
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
    test_non_finite();
    test_fp_enroll_stamps_and_roundtrips();
    test_fp_enroll_rules();
    test_fp_check();
    test_fp_registry_check();
    test_fp_v1_compat();
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_speaker_registry: PASS\n");
    return 0;
}
