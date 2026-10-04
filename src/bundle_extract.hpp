#pragma once
// A standalone single-model GGUF made from one component of a bundle, for a
// third-party loader that only opens a file by path (ced.cpp, voice-detect.cpp).
//
// The component's keys and tensors are written without the "<component>." prefix,
// so the result is a valid single-model GGUF that the loader reads as if it had
// been published on its own. Tensor data is streamed from the bundle in blocks;
// only the requested component's bytes are read.
//
// Where the file lives:
//   * Linux: an anonymous in-memory file (memfd_create), opened through
//     /proc/self/fd/N. Nothing is written to disk.
//   * Elsewhere, or when memfd_create or /proc is not available, or when the
//     environment variable PARAKEET_BUNDLE_NO_MEMFD is set to a value other than
//     "0": a temporary file in the system temporary directory (TMPDIR, TEMP,
//     else /tmp), created owner-only and removed when the ComponentFile is
//     destroyed. A crash between creation and removal leaves the file behind.
//
// Keep the object alive until the loader has finished opening the file, then
// let it go out of scope.
#include <cstdint>
#include <memory>
#include <string>

namespace pk {

class ComponentFile {
public:
    // Returns nullptr and writes the reason to *err on failure (bundle unreadable,
    // component missing, truncated bundle, no space for the file).
    static std::unique_ptr<ComponentFile> create(const std::string& bundle, const std::string& component,
                                                 std::string* err);
    ~ComponentFile();
    ComponentFile(const ComponentFile&) = delete;
    ComponentFile& operator=(const ComponentFile&) = delete;

    // Path to pass to a path-only loader. Valid until the object is destroyed.
    const std::string& path() const { return path_; }
    // True when the file is an in-memory file, false for a temporary file on disk.
    bool in_memory() const { return in_memory_; }
    // Size of the standalone GGUF in bytes.
    uint64_t size() const { return size_; }

private:
    ComponentFile() = default;
    std::string path_;
    bool in_memory_ = false;
    int fd_ = -1;            // memfd, when in_memory_
    bool remove_ = false;    // temporary file to delete
    uint64_t size_ = 0;
};

}  // namespace pk
