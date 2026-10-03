// A single bounded native-Linux file/ownership contract. No CUDA/DLL linkage.
#include "ddo_vmm.h"
#include "posix_file_io.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Workspace {
    std::filesystem::path path;
    Workspace() {
        auto pattern = (std::filesystem::temp_directory_path() / "ddo-source-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        char* created = ::mkdtemp(name.data());
        require(created != nullptr, "mkdtemp failed");
        path = created;
    }
    ~Workspace() { std::error_code error; std::filesystem::remove_all(path, error); }
};
struct Source {
    ddo_vmm_source_file* handle = nullptr;
    ~Source() { ddo_vmm_source_file_close(&handle); }
};
void write(const std::filesystem::path& path, const std::vector<unsigned char>& bytes) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(file.good(), "fixture write failed");
}
ddo_vmm_source_file_info info(Source& source) {
    ddo_vmm_source_file_info result{};
    result.version = 1;
    result.struct_size = sizeof(result);
    require(ddo_vmm_source_file_get_info(source.handle, &result) == DDO_VMM_SUCCESS, "metadata failed");
    return result;
}
}

int main() {
    try {
        Workspace workspace;
        auto path = workspace.path / "weights-unicode-\u00e9.bin";
        std::vector<unsigned char> bytes(3 * 1024 * 1024 + 19);
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<unsigned char>(i % 251);
        write(path, bytes);
        Source first, retained;
        require(ddo_vmm_source_file_open(path.c_str(), &first.handle) == DDO_VMM_SUCCESS, "open failed");
        const auto original = info(first);
        require(original.size_bytes == bytes.size() && original.generation != 0, "invalid file info");
        require(ddo_vmm_source_file_retain(first.handle, &retained.handle) == DDO_VMM_SUCCESS, "retain failed");
        require(ddo_vmm_source_file_close(&first.handle) == DDO_VMM_SUCCESS && !first.handle, "close failed");
        require(info(retained).generation == original.generation, "retained identity changed");
        std::vector<unsigned char> output(bytes.size());
        require(ddo_vmm_source_file_read(retained.handle, 0, output.data(), output.size()) == DDO_VMM_SUCCESS
            && output == bytes, "chunked read differs");
        require(ddo_vmm_source_file_read(retained.handle, bytes.size(), output.data(), 1) == DDO_VMM_INVALID_ARGUMENT,
            "out-of-file read admitted");
        require(ddo_vmm_source_file_read(retained.handle, UINT64_MAX, output.data(), 1) == DDO_VMM_INVALID_ARGUMENT,
            "overflowing read admitted");
        std::vector<std::future<void>> reads;
        for (uint64_t offset = 0; offset < 8; ++offset) {
            reads.push_back(std::async(std::launch::async, [&, offset] {
                std::array<unsigned char, 64> slice{};
                require(ddo_vmm_source_file_read(retained.handle, offset, slice.data(), slice.size()) == DDO_VMM_SUCCESS,
                    "concurrent positioned read failed");
                require(std::equal(slice.begin(), slice.end(), bytes.begin() + offset), "positioned reads interfere");
            }));
        }
        for (auto& read : reads) read.get();
        std::filesystem::remove(path);
        write(path, std::vector<unsigned char>{7, 8});
        Source replacement;
        require(ddo_vmm_source_file_open(path.c_str(), &replacement.handle) == DDO_VMM_SUCCESS, "replacement open failed");
        require(info(replacement).generation != original.generation, "replacement reused generation");
        require(ddo_vmm_source_file_read(retained.handle, 0, output.data(), 1) == DDO_VMM_SUCCESS
            && output[0] == bytes[0], "retained inode lost after pathname replacement");
        write(path, std::vector<unsigned char>{1, 2, 3});
        require(ddo_vmm_source_file_read(replacement.handle, 0, output.data(), 1) == DDO_VMM_SOURCE_CHANGED,
            "changed source metadata accepted");
        Source invalid;
        require(ddo_vmm_source_file_open(workspace.path.c_str(), &invalid.handle) == DDO_VMM_IO_ERROR
            && !invalid.handle, "directory admitted");
        auto fifo = workspace.path / "pipe";
        require(::mkfifo(fifo.c_str(), 0600) == 0, "FIFO fixture failed");
        require(ddo_vmm_source_file_open(fifo.c_str(), &invalid.handle) == DDO_VMM_IO_ERROR
            && !invalid.handle, "FIFO admitted or blocked");
        std::string error;
        int fd = ddo_source_io::open_file(path.c_str(), error);
        require(fd >= 0 && (::fcntl(fd, F_GETFD) & FD_CLOEXEC), "descriptor leaks across exec");
        require(!ddo_source_io::read_file(fd, 0, output.data(), 4, error), "short read accepted");
        require(!ddo_source_io::read_file(fd, UINT64_MAX, output.data(), 1, error), "off_t overflow accepted");
        ::close(fd);
        std::puts("POSIX source CPU contract PASS (no CUDA)");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "POSIX source CPU contract FAIL: %s\n", error.what());
        return 1;
    }
}
