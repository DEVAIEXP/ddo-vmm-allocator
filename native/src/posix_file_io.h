#pragma once

// Linux positioned file primitives. No CUDA, storage cache or worker ownership.
#if defined(__linux__)
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace ddo_source_io {
using file_handle = int;
constexpr file_handle invalid_file_handle = -1;

inline file_handle open_file(const char* path, std::string& error) {
    if (!path || !*path) {
        error = "native source path is required";
        return invalid_file_handle;
    }
    // NONBLOCK prevents a FIFO/device pathname from hanging before inspection.
    // Regular-file positioned reads ignore it; descriptors never cross exec.
    int fd;
    do { fd = ::open(path, O_RDONLY | O_CLOEXEC | O_NONBLOCK); }
    while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        error = "native source open failed: " + std::string(std::strerror(errno));
        return invalid_file_handle;
    }
    struct stat metadata{};
    int status;
    do { status = ::fstat(fd, &metadata); } while (status < 0 && errno == EINTR);
    if (status < 0 || !S_ISREG(metadata.st_mode) || metadata.st_size < 0) {
        error = "native source must be a readable regular file";
        // Linux close must not be retried on EINTR: the fd may already be reused.
        ::close(fd);
        return invalid_file_handle;
    }
    return fd;
}

inline bool read_file(file_handle fd, uint64_t offset, void* destination,
                      size_t size, std::string& error) {
    constexpr auto max_offset = static_cast<uint64_t>(std::numeric_limits<off_t>::max());
    if (fd < 0 || !destination || !size || offset > max_offset || size > max_offset - offset) {
        error = "native positioned read arguments exceed supported file offsets";
        return false;
    }
    auto* bytes = static_cast<unsigned char*>(destination);
    while (size) {
        const size_t chunk = std::min(size, size_t{2 * 1024 * 1024});
        ssize_t count;
        do { count = ::pread(fd, bytes, chunk, static_cast<off_t>(offset)); }
        while (count < 0 && errno == EINTR);
        if (count <= 0) {
            error = count == 0 ? "native positioned read was short (EOF)"
                : "native positioned read failed: " + std::string(std::strerror(errno));
            return false;
        }
        bytes += count;
        offset += static_cast<uint64_t>(count);
        size -= static_cast<size_t>(count);
    }
    return true;
}
}  // namespace ddo_source_io
#endif
