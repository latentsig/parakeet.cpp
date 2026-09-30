// Unit test for pk::write_file_atomic. No model needed.
#include "common.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__);  \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static std::string read_all(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

int main() {
    const fs::path dir = fs::temp_directory_path() / "pk_test_write_atomic";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir);
    const fs::path file = dir / "reg.bin";
    const std::string tmp = file.string() + ".tmp";

    // A new file gets exactly the bytes, including a NUL byte.
    const std::string first("first\0bytes", 11);
    std::string err;
    CHECK(pk::write_file_atomic(file.string(), first, &err));
    CHECK(err.empty());
    CHECK(read_all(file) == first);
    CHECK(!fs::exists(tmp));

    // Overwriting replaces the whole file and leaves no tmp file.
    const std::string second = "second, longer than the first one";
    CHECK(pk::write_file_atomic(file.string(), second, &err));
    CHECK(read_all(file) == second);
    CHECK(!fs::exists(tmp));
    CHECK(pk::write_file_atomic(file.string(), "x", nullptr));   // err may be null
    CHECK(read_all(file) == "x");
    CHECK(pk::write_file_atomic(file.string(), second, nullptr));

    // A path in a directory that does not exist fails with a reason, and the
    // existing file is untouched.
    err.clear();
    CHECK(!pk::write_file_atomic((dir / "missing" / "reg.bin").string(), "new", &err));
    CHECK(!err.empty());
    CHECK(read_all(file) == second);
    CHECK(!fs::exists(dir / "missing"));

    // A target that is a directory fails, the directory survives, no tmp file.
    const fs::path sub = dir / "sub";
    fs::create_directories(sub);
    err.clear();
    CHECK(!pk::write_file_atomic(sub.string(), "new", &err));
    CHECK(!err.empty());
    CHECK(fs::is_directory(sub));
    CHECK(!fs::exists(sub.string() + ".tmp"));
    CHECK(read_all(file) == second);

    // An empty path is refused.
    err.clear();
    CHECK(!pk::write_file_atomic("", "x", &err));
    CHECK(!err.empty());

    fs::remove_all(dir, ec);
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test_write_atomic: PASS\n");
    return 0;
}
