#include "bundle_extract.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

#include "ggml.h"
#include "gguf.h"

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#elif !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace pk {

namespace {

#if defined(_WIN32)
typedef long long off_t_compat;
#define PK_FSEEK _fseeki64
#define PK_FTELL _ftelli64
#else
typedef off_t off_t_compat;
#define PK_FSEEK fseeko
#define PK_FTELL ftello
#endif

void set_err(std::string* err, const std::string& m) {
    if (err) *err = m;
}

// Copies one key of `g` to `out` under `name`. Unknown value types are skipped.
void copy_kv(gguf_context* out, gguf_context* g, int64_t i, const std::string& name) {
    const char* k = name.c_str();
    switch (gguf_get_kv_type(g, i)) {
        case GGUF_TYPE_UINT8:   gguf_set_val_u8(out, k, gguf_get_val_u8(g, i)); break;
        case GGUF_TYPE_INT8:    gguf_set_val_i8(out, k, gguf_get_val_i8(g, i)); break;
        case GGUF_TYPE_UINT16:  gguf_set_val_u16(out, k, gguf_get_val_u16(g, i)); break;
        case GGUF_TYPE_INT16:   gguf_set_val_i16(out, k, gguf_get_val_i16(g, i)); break;
        case GGUF_TYPE_UINT32:  gguf_set_val_u32(out, k, gguf_get_val_u32(g, i)); break;
        case GGUF_TYPE_INT32:   gguf_set_val_i32(out, k, gguf_get_val_i32(g, i)); break;
        case GGUF_TYPE_FLOAT32: gguf_set_val_f32(out, k, gguf_get_val_f32(g, i)); break;
        case GGUF_TYPE_UINT64:  gguf_set_val_u64(out, k, gguf_get_val_u64(g, i)); break;
        case GGUF_TYPE_INT64:   gguf_set_val_i64(out, k, gguf_get_val_i64(g, i)); break;
        case GGUF_TYPE_FLOAT64: gguf_set_val_f64(out, k, gguf_get_val_f64(g, i)); break;
        case GGUF_TYPE_BOOL:    gguf_set_val_bool(out, k, gguf_get_val_bool(g, i)); break;
        case GGUF_TYPE_STRING:  gguf_set_val_str(out, k, gguf_get_val_str(g, i)); break;
        case GGUF_TYPE_ARRAY: {
            const gguf_type at = gguf_get_arr_type(g, i);
            const size_t n = gguf_get_arr_n(g, i);
            if (at == GGUF_TYPE_STRING) {
                std::vector<const char*> v(n);
                for (size_t j = 0; j < n; ++j) v[j] = gguf_get_arr_str(g, i, j);
                gguf_set_arr_str(out, k, v.data(), n);
            } else if (at != GGUF_TYPE_ARRAY) {
                gguf_set_arr_data(out, k, at, gguf_get_arr_data(g, i), n);
            }
            break;
        }
        default: break;
    }
}

bool no_memfd_requested() {
    const char* e = std::getenv("PARAKEET_BUNDLE_NO_MEMFD");
    return e && *e && std::strcmp(e, "0") != 0;
}

// Writes the standalone GGUF of `comp` to `f` (a freshly opened, empty file).
// Returns the number of bytes written, or 0 with *err set.
uint64_t write_component(const std::string& bundle, const std::string& comp, FILE* f, std::string* err) {
    ggml_context* meta = nullptr;
    gguf_init_params ip{/*no_alloc*/ true, &meta};
    gguf_context* g = gguf_init_from_file(bundle.c_str(), ip);
    if (!g) {
        set_err(err, "cannot read " + bundle + " as a GGUF file");
        return 0;
    }
    const std::string pre = comp + ".";
    gguf_context* out = gguf_init_empty();
    ggml_context* tctx = nullptr;
    FILE* in = nullptr;
    uint64_t written = 0;
    bool ok = false;
    do {
        for (int64_t i = 0; i < gguf_get_n_kv(g); ++i) {
            const char* k = gguf_get_key(g, i);
            if (std::strncmp(k, pre.c_str(), pre.size()) != 0) continue;
            copy_kv(out, g, i, std::string(k + pre.size()));
        }
        const int64_t nt = gguf_get_n_tensors(g);
        std::vector<int64_t> ids;
        for (int64_t i = 0; i < nt; ++i)
            if (std::strncmp(gguf_get_tensor_name(g, i), pre.c_str(), pre.size()) == 0) ids.push_back(i);
        if (ids.empty()) {
            set_err(err, "bundle " + bundle + ": component '" + comp + "' has no tensors");
            break;
        }
        // Tensor descriptors only (no data): the writer needs type, shape and name.
        ggml_init_params tp{ggml_tensor_overhead() * ids.size() + (1u << 16), nullptr, /*no_alloc*/ true};
        tctx = ggml_init(tp);
        if (!tctx) { set_err(err, "out of memory"); break; }
        bool bad = false;
        for (int64_t i : ids) {
            const char* nm = gguf_get_tensor_name(g, i);
            ggml_tensor* s = ggml_get_tensor(meta, nm);
            ggml_tensor* t = s ? ggml_new_tensor(tctx, s->type, GGML_MAX_DIMS, s->ne) : nullptr;
            if (!t) { set_err(err, std::string("cannot describe tensor ") + nm); bad = true; break; }
            ggml_set_name(t, nm + pre.size());
            gguf_add_tensor(out, t);
        }
        if (bad) break;
        if (!gguf_write_to_file_ptr(out, f, /*only_meta*/ true)) { set_err(err, "cannot write the component file"); break; }
        written = (uint64_t)gguf_get_data_offset(out);   // header + tensor table, padded
        in = std::fopen(bundle.c_str(), "rb");
        if (!in) { set_err(err, "cannot open " + bundle); break; }
        PK_FSEEK(in, 0, SEEK_END);
        const uint64_t file_size = (uint64_t)PK_FTELL(in);
        const uint64_t base = (uint64_t)gguf_get_data_offset(g);
        std::vector<char> buf(1u << 20);
        bool failed = false;
        for (size_t n = 0; n < ids.size() && !failed; ++n) {
            const int64_t i = ids[n];
            const uint64_t size = gguf_get_tensor_size(g, i);
            const uint64_t off = base + gguf_get_tensor_offset(g, i);
            if (off + size > file_size) {
                set_err(err, std::string("bundle ") + bundle + " is truncated (tensor " + gguf_get_tensor_name(g, i) + ")");
                failed = true;
                break;
            }
            // Pad up to this tensor's aligned offset in the output.
            const uint64_t want = (uint64_t)gguf_get_data_offset(out) + gguf_get_tensor_offset(out, (int64_t)n);
            while (written < want) {
                const char zero[64] = {0};
                const size_t k = (size_t)std::min<uint64_t>(sizeof zero, want - written);
                if (std::fwrite(zero, 1, k, f) != k) { failed = true; break; }
                written += k;
            }
            if (failed || PK_FSEEK(in, (off_t_compat)off, SEEK_SET) != 0) { failed = true; break; }
            for (uint64_t done = 0; done < size && !failed;) {
                const size_t k = (size_t)std::min<uint64_t>(buf.size(), size - done);
                if (std::fread(buf.data(), 1, k, in) != k || std::fwrite(buf.data(), 1, k, f) != k) failed = true;
                done += k;
                written += k;
            }
        }
        if (failed) {
            if (!err || err->empty()) set_err(err, "cannot copy the component data (disk or memory full?)");
            break;
        }
        // Pad the end to the alignment like the single-pass writer does.
        const size_t al = gguf_get_alignment(out);
        while (written % al) {
            if (std::fputc(0, f) == EOF) { failed = true; break; }
            ++written;
        }
        if (failed || std::fflush(f) != 0) { set_err(err, "cannot write the component file"); break; }
        ok = true;
    } while (false);
    if (in) std::fclose(in);
    if (tctx) ggml_free(tctx);
    gguf_free(out);
    gguf_free(g);
    if (meta) ggml_free(meta);
    return ok ? written : 0;
}

}  // namespace

ComponentFile::~ComponentFile() {
#if !defined(_WIN32)
    if (fd_ >= 0) ::close(fd_);
#endif
    if (remove_) {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
}

std::unique_ptr<ComponentFile> ComponentFile::create(const std::string& bundle, const std::string& comp,
                                                     std::string* err) {
    std::unique_ptr<ComponentFile> cf(new ComponentFile());
#if defined(__linux__)
    if (!no_memfd_requested()) {
        const int fd = (int)memfd_create("parakeet-component", MFD_CLOEXEC);
        if (fd >= 0) {
            const std::string p = "/proc/self/fd/" + std::to_string(fd);
            FILE* f = ::fdopen(::dup(fd), "wb");
            std::string e;
            const uint64_t n = f ? write_component(bundle, comp, f, &e) : 0;
            if (f) std::fclose(f);
            struct stat st {};
            if (n > 0 && ::stat(p.c_str(), &st) == 0) {   // /proc must be mounted for the path to work
                cf->fd_ = fd;
                cf->path_ = p;
                cf->in_memory_ = true;
                cf->size_ = n;
                return cf;
            }
            ::close(fd);
            // A real failure (missing component, truncated bundle) is final; only fall back
            // when the in-memory file itself could not be used.
            if (n == 0 && !e.empty() && e.find("cannot write") == std::string::npos &&
                e.find("cannot copy") == std::string::npos) {
                set_err(err, e);
                return nullptr;
            }
        }
    }
#endif
    // Temporary file.
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec) { set_err(err, "no temporary directory for the component file"); return nullptr; }
    FILE* f = nullptr;
    std::string path;
#if defined(_WIN32)
    for (int attempt = 0; attempt < 100 && !f; ++attempt) {
        path = (dir / ("parakeet-component-" + std::to_string(std::rand()) + "-" + std::to_string(attempt) + ".gguf")).string();
        if (std::filesystem::exists(path)) continue;
        f = std::fopen(path.c_str(), "wb");
    }
#else
    for (int attempt = 0; attempt < 100 && !f; ++attempt) {
        path = (dir / ("parakeet-component-" + std::to_string((long)::getpid()) + "-" + std::to_string(std::rand()) +
                       "-" + std::to_string(attempt) + ".gguf")).string();
        const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);   // never follow or reuse a file
        if (fd >= 0) f = ::fdopen(fd, "wb");
    }
#endif
    if (!f) { set_err(err, "cannot create a temporary file in " + dir.string()); return nullptr; }
    cf->path_ = path;
    cf->remove_ = true;   // from here the destructor deletes the file, also on failure
    std::string e;
    const uint64_t n = write_component(bundle, comp, f, &e);
    std::fclose(f);
    if (n == 0) { set_err(err, e.empty() ? "cannot write the temporary file " + path : e); return nullptr; }
    cf->size_ = n;
    return cf;
}

}  // namespace pk
