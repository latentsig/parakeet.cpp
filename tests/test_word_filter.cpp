// pk::apply_word_filter on scripted words and confidences (no model).
#include "transcription.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

using namespace pk;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL: %s (line %d)\n", #c, __LINE__); ++failures; } } while (0)

struct W { const char* text; float start; float conf; };

// One token per word: token i has id i and belongs to word i.
static Transcription make(const std::vector<W>& ws) {
    Transcription t;
    for (size_t i = 0; i < ws.size(); ++i) {
        Word w;
        w.text = ws[i].text;
        w.start = ws[i].start;
        w.end = w.start + 0.2f;
        w.conf = ws[i].conf;
        w.tok_first = w.tok_last = (int32_t)i;
        t.words.push_back(w);
        t.tokens.push_back(TokenInfo{(int32_t)i, (int32_t)(w.start / 0.08f), w.conf, 1});
        if (!t.text.empty()) t.text += ' ';
        t.text += w.text;
    }
    return t;
}

static bool same(const Transcription& a, const Transcription& b) {
    if (a.text != b.text || a.words.size() != b.words.size() || a.tokens.size() != b.tokens.size() ||
        a.dropped_words != b.dropped_words) return false;
    for (size_t i = 0; i < a.words.size(); ++i) {
        const Word &x = a.words[i], &y = b.words[i];
        if (x.text != y.text || x.start != y.start || x.end != y.end || x.conf != y.conf ||
            x.tok_first != y.tok_first || x.tok_last != y.tok_last) return false;
    }
    for (size_t i = 0; i < a.tokens.size(); ++i)
        if (a.tokens[i].id != b.tokens[i].id || a.tokens[i].frame != b.tokens[i].frame ||
            a.tokens[i].conf != b.tokens[i].conf || a.tokens[i].span != b.tokens[i].span) return false;
    return true;
}

static void test_off_is_identity() {
    const Transcription ref = make({{"a", 0.f, 0.9f}, {"b", 1.f, 0.05f}, {".", 2.f, 0.1f}});
    Transcription t = ref;
    CHECK(!WordFilter().active());
    CHECK(WordFilter().min_local_conf == 0.0f && WordFilter().local_radius_sec == 5.0f && !WordFilter().drop_punct_only);
    CHECK(apply_word_filter(t, WordFilter()) == 0);
    CHECK(same(t, ref) && t.dropped_words == -1);
    // A radius alone does not turn the filter on.
    WordFilter f;
    f.local_radius_sec = 1.0f;
    CHECK(!f.active() && apply_word_filter(t, f) == 0 && same(t, ref));
    // Empty input.
    Transcription e;
    f.min_local_conf = 0.5f;
    CHECK(apply_word_filter(e, f) == 0 && e.words.empty() && e.dropped_words == -1);
}

static void test_isolated_low_word_between_confident_words_is_kept() {
    Transcription t = make({{"a", 0.0f, 1.0f}, {"b", 0.3f, 0.2f}, {"c", 0.6f, 1.0f}});
    WordFilter f;
    f.min_local_conf = 0.5f;
    CHECK(apply_word_filter(t, f) == 0);
    CHECK(t.words.size() == 3 && t.text == "a b c" && t.tokens.size() == 3);
    CHECK(t.dropped_words == 0);  // the filter ran and dropped nothing
}

static void test_run_of_low_words_is_dropped() {
    // Speech, then a run of low confidence words 20 s later, then speech.
    Transcription t = make({{"hello", 0.0f, 0.95f}, {"world", 0.5f, 0.9f},
                            {"x", 20.0f, 0.2f}, {"y", 20.4f, 0.3f}, {"z", 20.8f, 0.25f},
                            {"again", 40.0f, 0.9f}});
    WordFilter f;
    f.min_local_conf = 0.5f;
    CHECK(apply_word_filter(t, f) == 3);
    CHECK(t.dropped_words == 3);
    CHECK(t.text == "hello world again");
    CHECK(t.words.size() == 3 && t.tokens.size() == 3);
    CHECK(t.tokens[0].id == 0 && t.tokens[1].id == 1 && t.tokens[2].id == 5);
    CHECK(t.words[2].text == "again" && t.words[2].tok_first == 2 && t.words[2].tok_last == 2);
    // A lone low confidence word with no neighbour in the radius is dropped too.
    t = make({{"a", 0.0f, 0.9f}, {"lone", 30.0f, 0.4f}});
    CHECK(apply_word_filter(t, f) == 1 && t.text == "a");
}

static void test_radius_and_boundaries() {
    WordFilter f;
    f.min_local_conf = 0.5f;
    f.local_radius_sec = 5.0f;
    // B is exactly 5 s after A: inside the window (the bound is inclusive), so
    // B's mean is 0.6 and it stays.
    Transcription t = make({{"A", 0.0f, 1.0f}, {"B", 5.0f, 0.2f}});
    CHECK(apply_word_filter(t, f) == 0);
    // Just under 5 s of radius: B is alone, mean 0.2, dropped. A keeps its own 1.0.
    f.local_radius_sec = 4.99f;
    t = make({{"A", 0.0f, 1.0f}, {"B", 5.0f, 0.2f}});
    CHECK(apply_word_filter(t, f) == 1 && t.text == "A");
    // The window looks both ways: a low word followed by confident words stays.
    f.local_radius_sec = 5.0f;
    t = make({{"B", 0.0f, 0.2f}, {"A", 1.0f, 1.0f}, {"C", 2.0f, 1.0f}});
    CHECK(apply_word_filter(t, f) == 0);
    // A threshold equal to the mean keeps the words (strictly below is dropped).
    f.min_local_conf = 0.75f;
    t = make({{"A", 0.0f, 1.0f}, {"B", 1.0f, 0.5f}});  // both have mean 0.75
    CHECK(apply_word_filter(t, f) == 0);
    f.min_local_conf = 0.76f;
    t = make({{"A", 0.0f, 1.0f}, {"B", 1.0f, 0.5f}});
    CHECK(apply_word_filter(t, f) == 2 && t.words.empty() && t.text.empty() && t.tokens.empty());
    // Words out of time order give the same answer as sorted ones.
    f.min_local_conf = 0.5f;
    f.local_radius_sec = 2.0f;
    t = make({{"x", 20.0f, 0.2f}, {"a", 0.0f, 0.9f}, {"y", 20.5f, 0.3f}, {"b", 0.5f, 0.9f}});
    CHECK(apply_word_filter(t, f) == 2 && t.text == "a b");
}

static void test_punct_only() {
    WordFilter f;
    f.drop_punct_only = true;
    CHECK(f.active());
    Transcription t = make({{"a.", 0.0f, 0.9f}, {".", 1.0f, 0.9f}, {"?!", 2.0f, 0.9f}, {"...", 3.0f, 0.9f},
                            {"\xE2\x80\xA6", 4.0f, 0.9f},      // ellipsis
                            {"\xE2\x80\x94", 5.0f, 0.9f},      // em dash
                            {"\xC2\xBF", 6.0f, 0.9f},          // inverted question mark
                            {"1", 7.0f, 0.9f}, {"\xC3\xA9", 8.0f, 0.9f},  // e acute is a letter
                            {"\xE6\x97\xA5", 9.0f, 0.9f},      // a CJK letter
                            {"-5", 10.0f, 0.9f}});
    CHECK(apply_word_filter(t, f) == 6);
    CHECK(t.words.size() == 5);
    CHECK(t.text == "a. 1 \xC3\xA9 \xE6\x97\xA5 -5");
    CHECK(t.tokens.size() == 5 && t.tokens[1].id == 7 && t.words[1].tok_first == 1);
    // The local filter does not touch confident punctuation, and punct-only
    // words count in the mean (that is the input the decoder gave).
    f.drop_punct_only = false;
    f.min_local_conf = 0.5f;
    t = make({{".", 0.0f, 0.9f}, {"a", 1.0f, 0.9f}});
    CHECK(apply_word_filter(t, f) == 0);
}

static void test_both_checks_together_and_accumulate() {
    WordFilter f;
    f.min_local_conf = 0.5f;
    f.drop_punct_only = true;
    Transcription t = make({{"a", 0.0f, 0.9f}, {".", 0.5f, 0.9f}, {"n", 30.0f, 0.1f}});
    CHECK(apply_word_filter(t, f) == 2 && t.text == "a" && t.dropped_words == 2);
    // A second call on the same transcription adds to the count.
    Word q;
    q.text = "q"; q.start = 30.0f; q.end = 30.2f; q.conf = 0.1f; q.tok_first = q.tok_last = (int32_t)t.tokens.size();
    t.words.push_back(q);
    t.tokens.push_back(TokenInfo{9, 12, 0.1f, 1});
    t.text += " q";
    CHECK(apply_word_filter(t, f) == 1 && t.dropped_words == 3 && t.text == "a");
}

static void test_group_words_records_token_ranges() {
    // "hello", "world," (two pieces and a comma), "hi".
    const std::vector<std::string> pieces = {"\xE2\x96\x81hello", "\xE2\x96\x81wor", "ld", ",", "\xE2\x96\x81hi"};
    std::vector<TokenInfo> tokens;
    for (int i = 0; i < 5; ++i) tokens.push_back(TokenInfo{i, i * 2, 0.9f, 1});
    const std::vector<Word> ws = group_words(tokens, pieces, 0.08f);
    CHECK(ws.size() == 3);
    if (ws.size() == 3) {
        CHECK(ws[0].text == "hello" && ws[0].tok_first == 0 && ws[0].tok_last == 0);
        CHECK(ws[1].text == "world," && ws[1].tok_first == 1 && ws[1].tok_last == 3);
        CHECK(ws[2].text == "hi" && ws[2].tok_first == 4 && ws[2].tok_last == 4);
    }
    // Dropping the middle word removes its three tokens and renumbers the last.
    Transcription t;
    t.words = ws;
    t.tokens = tokens;
    t.words[1].conf = 0.05f;
    t.words[0].start = 0.0f; t.words[1].start = 30.0f; t.words[2].start = 60.0f;
    for (Word& w : t.words) { w.conf = 0.9f; }
    t.words[1].conf = 0.05f;
    t.text = "hello world, hi";
    WordFilter f;
    f.min_local_conf = 0.5f;
    CHECK(apply_word_filter(t, f) == 1);
    CHECK(t.text == "hello hi" && t.tokens.size() == 2 && t.tokens[0].id == 0 && t.tokens[1].id == 4);
    CHECK(t.words[1].tok_first == 1 && t.words[1].tok_last == 1);
}

int main() {
    test_off_is_identity();
    test_isolated_low_word_between_confident_words_is_kept();
    test_run_of_low_words_is_dropped();
    test_radius_and_boundaries();
    test_punct_only();
    test_both_checks_together_and_accumulate();
    test_group_words_records_token_ranges();
    if (failures) return 1;
    std::puts("test_word_filter: OK");
    return 0;
}
