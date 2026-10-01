#include "speaker_identifier.hpp"
#include "parakeet_capi.h"
#include "speaker_model_identity.hpp"
#include "speaker_profiles_json.hpp"
#include <fstream>
#include <locale>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

static int failures = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #x); ++failures; } } while (0)
struct CommaDecimal : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
};
int main() {
    // Published SHA-256 known answers: empty, abc, multiblock, million a's.
    const char* file = "speaker-profile-hash-test.bin";
    std::vector<std::pair<std::string, std::string>> vectors{
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"}};
    vectors.push_back({std::string(55, 'a'), "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"});
    vectors.push_back({std::string(56, 'a'), "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"});
    vectors.push_back({std::string(63, 'a'), "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34"});
    vectors.push_back({std::string(64, 'a'), "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"});
    vectors.push_back({std::string(65, 'a'), "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0"});
    vectors.push_back({std::string(65535, 'a'), "6e1bebca6a8229364a162a72ef064826c4cd7457bf54f190ef782bd9deff3e42"});
    vectors.push_back({std::string(65536, 'a'), "bf718b6f653bebc184e1479f1935b8da974d701b893afcf49e701f3e2f9f9c5a"});
    vectors.push_back({std::string(65537, 'a'), "008ffc88d3c96a9f307524eb361e47c5222a887fc45fa0c1fb8d429c5c23b430"});
    for (const auto& v : vectors) {
        { std::ofstream f(file, std::ios::binary); f << v.first; }
        CHECK(pk::speaker_model_identity(file) == "sha256:" + v.second);
    }
    // Hash before and after the actual load, rejecting changed or removed files.
    std::string identity;
    auto loaded = pk::load_speaker_with_identity(file, identity, [] { return 42; });
    CHECK(loaded == 42 && identity == "sha256:" + vectors.back().second);
    bool changed = false;
    try {
        pk::load_speaker_with_identity(file, identity, [&] {
            std::ofstream f(file); f << "replaced during load"; return 42;
        });
    } catch (const std::runtime_error&) { changed = true; }
    CHECK(changed && identity.empty());
    std::remove(file);
    bool missing = false;
    try { pk::speaker_model_identity(file); } catch (const std::runtime_error&) { missing = true; }
    CHECK(missing);

    pk::SpeakerRegistry reg;
    pk::SpeakerIdOpts opts;
    std::vector<float> pcm(8 * 16000, 0.1f);
    // Slot 0 has 3 clean seconds, slot 1 has only one; slot 2 is fully overlapped.
    std::vector<pk::SpeakerSegment> segs{{0,0,4}, {1,3,5}, {2,3,4}};
    int calls = 0;
    pk::SpeakerEmbed embed = [&](const float*, int n, std::vector<float>& out) {
        ++calls; CHECK(n == 3 * 16000); out = {3,4}; return true;
    };
    std::map<int, pk::SpeakerProfile> profiles;
    auto names = pk::identify_offline(pcm, segs, embed, reg, opts, &profiles, 2);
    CHECK(names.size() == 3 && names.at(0).name.empty());
    CHECK(calls == 1 && profiles.size() == 3);
    CHECK(profiles.at(0).unavailable_reason.empty());
    CHECK(profiles.at(0).clean_duration == 3);
    CHECK(profiles.at(0).intervals.size() == 1);
    CHECK(profiles.at(0).intervals[0].start == 0 && profiles.at(0).intervals[0].end == 3);
    CHECK(std::fabs(profiles.at(0).embedding[0] - 0.6f) < 1e-6);
    CHECK(profiles.at(1).clean_duration == 1);
    CHECK(profiles.at(1).unavailable_reason == "insufficient_clean_speech");
    CHECK(profiles.at(2).clean_duration == 0);
    CHECK(profiles.at(2).embedding.empty());
    // The production public JSON assembler, without requiring model inference.
    const std::string base = "{\"speakers\":3,\"segments\":[{\"speaker\":0,\"start\":0.00,\"end\":4.00}]}";
    const std::string nj = "{\"0\":{\"name\":\"\",\"score\":0.0000}}";
    const auto legacy_json = pk::named_profiles_json(base, nj, nullptr, "", 0);
    CHECK(legacy_json == base.substr(0, base.size()-1) + ",\"names\":" + nj + "}");
    CHECK(legacy_json.find("embedding") == std::string::npos);
    CHECK(legacy_json.find("speaker_profiles") == std::string::npos);
    const auto prior_locale = std::locale();
    std::locale::global(std::locale(prior_locale, new CommaDecimal));
    const auto json = pk::named_profiles_json(base, nj, &profiles, "sha256:abc", 2);
    std::locale::global(prior_locale);
    CHECK(json.substr(0, legacy_json.size()-1) == legacy_json.substr(0, legacy_json.size()-1));
    CHECK(json.find("\"version\":1,\"encoder\":{\"identity\":\"sha256:abc\",\"dimension\":2}") != std::string::npos);
    CHECK(json.find("\"clean_duration\":3,\"intervals\":[{\"start\":0,\"end\":3}]") != std::string::npos);
    CHECK(json.find("\"unavailable_reason\":null,\"embedding\":[0.600000024,0.800000012]") != std::string::npos);
    CHECK(json.find("\"speaker\":2,\"clean_duration\":0,\"intervals\":[],\"unavailable_reason\":\"insufficient_clean_speech\"}") != std::string::npos);
    CHECK(json.find("embedding") == json.rfind("embedding"));
    CHECK(json.find("embedding") > json.find("speaker_profiles"));
    std::map<int, pk::SpeakerProfile> no_profiles;
    CHECK(pk::named_profiles_json(base, nj, &no_profiles, "sha256:abc", 2).find("\"speakers\":[]") != std::string::npos);
    // Export -> existing raw-vector C registration -> future matching.
    auto* cr = parakeet_capi_speaker_registry_new();
    const auto v = profiles.at(0).embedding;
    CHECK(parakeet_capi_speaker_registry_add_embedding(cr, "Alice", v.data(), (int)v.size()) == 0);
    CHECK(parakeet_capi_speaker_registry_size(cr) == 1);
    parakeet_capi_speaker_registry_free(cr);
    reg.enroll("Alice", v);
    names = pk::identify_offline(pcm, segs, embed, reg, opts, &profiles, 2);
    CHECK(names.at(0).name == "Alice");
    const auto legacy = pk::identify_offline(pcm, segs, embed, reg, opts);
    CHECK(legacy.at(0).name == names.at(0).name);
    CHECK(legacy.at(0).score == names.at(0).score);
    for (auto bad : std::vector<std::vector<float>>{{}, {0,0}, {1}, {1,2,3},
             {std::numeric_limits<float>::quiet_NaN(),1}, {1,std::numeric_limits<float>::infinity()}}) {
        pk::SpeakerEmbed invalid = [bad](const float*, int, std::vector<float>& out) { out = bad; return true; };
        pk::identify_offline(pcm, segs, invalid, reg, opts, &profiles, 2);
        CHECK(profiles.at(0).unavailable_reason == "invalid_embedding");
        CHECK(profiles.at(0).embedding.empty());
    }
    pk::SpeakerEmbed failed = [](const float*, int, std::vector<float>&) { return false; };
    pk::identify_offline(pcm, segs, failed, reg, opts, &profiles, 2);
    CHECK(profiles.at(0).unavailable_reason == "embedding_failed");
    bool threw = false;
    try { pk::identify_offline(pcm, segs, failed, reg, opts); } catch (const std::runtime_error&) { threw = true; }
    CHECK(threw); // legacy failure semantics do not change
    // Newest 30 seconds only, preview describes precisely the embedded samples.
    pcm.resize(40 * 16000);
    int length = 0;
    auto cap = [&](const float*, int n, std::vector<float>& out) { length = n; out = {1,0}; return true; };
    pk::identify_offline(pcm, {{0,0,20}, {0,21,40}}, cap, reg, opts, &profiles, 2);
    CHECK(length == 30 * 16000 && profiles.at(0).clean_duration == 30);
    CHECK(profiles.at(0).intervals.size() == 2);
    CHECK(profiles.at(0).intervals[0].start == 9 && profiles.at(0).intervals[1].start == 21);
    // Clipping, duplicate spans and unsorted input never duplicate voice.
    pk::identify_offline(pcm, {{0,21,50}, {0,-5,20}, {0,0,20}}, cap, reg, opts, &profiles, 2);
    CHECK(profiles.at(0).clean_duration == 30);
    CHECK(profiles.at(0).intervals[0].start == 9 && profiles.at(0).intervals[1].end == 40);
    CHECK(parakeet_capi_diarize_profiles_pcm_json(nullptr, nullptr, nullptr, nullptr, 0, 16000, 0, 0) == nullptr);
    CHECK(parakeet_capi_speaker_identity(nullptr) == nullptr);
    return failures ? 1 : 0;
}
