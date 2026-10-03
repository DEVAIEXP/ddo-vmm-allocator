#include "ddo_vmm.h"
#include "source_io.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <exception>
#include <memory>
#include <new>

namespace {
thread_local char source_error[512] = {};
std::atomic<uint64_t> next_generation{1};

ddo_vmm_status fail(ddo_vmm_status status, const char* message) noexcept {
    std::snprintf(source_error, sizeof(source_error), "%s", message);
    return status;
}

template<class Operation>
ddo_vmm_status boundary(Operation&& operation) noexcept {
    source_error[0] = '\0';
    try {
        return operation();
    } catch (const std::bad_alloc&) {
        return fail(DDO_VMM_OUT_OF_MEMORY, "file source allocation failed");
    } catch (const std::exception& error) {
        return fail(DDO_VMM_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(DDO_VMM_INTERNAL_ERROR, "unexpected file source failure");
    }
}

struct FileState {
#if defined(_WIN32)
    HANDLE handle = INVALID_HANDLE_VALUE;
    ~FileState() { if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle); }
#elif defined(__linux__)
    ddo_source_io::file_handle handle = ddo_source_io::invalid_file_handle;
    ~FileState() { if (handle >= 0) ::close(handle); }
#endif
    ddo_vmm_source_file_info info{};
};

#if defined(_WIN32)
uint64_t join_words(DWORD high, DWORD low) {
    return (static_cast<uint64_t>(high) << 32) | low;
}

ddo_vmm_status inspect(HANDLE handle, ddo_vmm_source_file_info& info) {
    BY_HANDLE_FILE_INFORMATION metadata{};
    if (GetFileType(handle) != FILE_TYPE_DISK || !GetFileInformationByHandle(handle, &metadata)
        || (metadata.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return fail(DDO_VMM_IO_ERROR, "file source must be a readable regular disk file");
    }
    info = {};
    info.struct_size = sizeof(info);
    info.version = 1;
    info.volume_id = metadata.dwVolumeSerialNumber;
    info.file_id = join_words(metadata.nFileIndexHigh, metadata.nFileIndexLow);
    info.size_bytes = join_words(metadata.nFileSizeHigh, metadata.nFileSizeLow);
    info.last_write_time = join_words(metadata.ftLastWriteTime.dwHighDateTime, metadata.ftLastWriteTime.dwLowDateTime);
    return DDO_VMM_SUCCESS;
}

#elif defined(__linux__)
ddo_vmm_status inspect(ddo_source_io::file_handle handle, ddo_vmm_source_file_info& info) {
    struct stat metadata{};
    int status;
    do { status = ::fstat(handle, &metadata); } while (status < 0 && errno == EINTR);
    if (status < 0 || !S_ISREG(metadata.st_mode) || metadata.st_size < 0)
        return fail(DDO_VMM_IO_ERROR, "file source must be a readable regular file");
    info = {};
    info.struct_size = sizeof(info);
    info.version = 1;
    info.volume_id = static_cast<uint64_t>(metadata.st_dev);
    info.file_id = static_cast<uint64_t>(metadata.st_ino);
    info.size_bytes = static_cast<uint64_t>(metadata.st_size);
    // Platform-local opaque timestamp, not interchangeable with Windows FILETIME.
    info.last_write_time = static_cast<uint64_t>(metadata.st_mtim.tv_sec) * 1000000000ULL
        + static_cast<uint64_t>(metadata.st_mtim.tv_nsec);
    return DDO_VMM_SUCCESS;
}
#endif

#if defined(_WIN32) || defined(__linux__)
ddo_vmm_status validate(const FileState& state) {
    ddo_vmm_source_file_info current{};
    auto status = inspect(state.handle, current);
    if (status != DDO_VMM_SUCCESS) return status;
    const auto& expected = state.info;
    if (current.volume_id != expected.volume_id || current.file_id != expected.file_id
        || current.size_bytes != expected.size_bytes || current.last_write_time != expected.last_write_time) {
        return fail(DDO_VMM_SOURCE_CHANGED, "immutable file source metadata changed");
    }
    return DDO_VMM_SUCCESS;
}
#endif
}

struct ddo_vmm_source_file {
    std::shared_ptr<FileState> state;
};

#if defined(_WIN32)
HANDLE ddo_source_io::source_handle(const ddo_vmm_source_file* source) {
    return source ? source->state->handle : INVALID_HANDLE_VALUE;
}
#elif defined(__linux__)
ddo_source_io::file_handle ddo_source_io::source_handle(const ddo_vmm_source_file* source) {
    return source ? source->state->handle : invalid_file_handle;
}
#endif

extern "C" uint32_t ddo_vmm_source_abi_version(void) { return 1; }
extern "C" const char* ddo_vmm_source_last_error(void) { return source_error; }

extern "C" ddo_vmm_status ddo_vmm_source_file_open(const char* path, ddo_vmm_source_file** out_file) {
    return boundary([&]() -> ddo_vmm_status {
        if (!out_file) return fail(DDO_VMM_INVALID_ARGUMENT, "source output handle is required");
        *out_file = nullptr;
        if (!path || !*path) return fail(DDO_VMM_INVALID_ARGUMENT, "source path is required");
#if defined(_WIN32) || defined(__linux__)
        auto state = std::make_shared<FileState>();
        std::string error;
        state->handle = ddo_source_io::open_file(path, error);
#if defined(_WIN32)
        if (state->handle == INVALID_HANDLE_VALUE) return fail(DDO_VMM_IO_ERROR, error.c_str());
#else
        if (state->handle == ddo_source_io::invalid_file_handle) return fail(DDO_VMM_IO_ERROR, error.c_str());
#endif
        auto status = inspect(state->handle, state->info);
        if (status != DDO_VMM_SUCCESS) return status;
        state->info.generation = next_generation.fetch_add(1, std::memory_order_relaxed);
        auto owner = std::make_unique<ddo_vmm_source_file>();
        owner->state = std::move(state);
        *out_file = owner.release();
        return DDO_VMM_SUCCESS;
#else
        return fail(DDO_VMM_UNSUPPORTED, "native file source requires Windows or Linux");
#endif
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_file_retain(
    const ddo_vmm_source_file* file, ddo_vmm_source_file** out_file
) {
    return boundary([&]() -> ddo_vmm_status {
        if (!out_file) return fail(DDO_VMM_INVALID_ARGUMENT, "source output handle is required");
        *out_file = nullptr;
        if (!file) return fail(DDO_VMM_INVALID_ARGUMENT, "source is closed");
        auto retained = std::make_unique<ddo_vmm_source_file>();
        retained->state = file->state;
        *out_file = retained.release();
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_file_get_info(
    const ddo_vmm_source_file* file, ddo_vmm_source_file_info* info
) {
    return boundary([&]() -> ddo_vmm_status {
        if (!file || !info || info->version != 1 || info->struct_size < sizeof(*info))
            return fail(DDO_VMM_INVALID_ARGUMENT, "source info requires an open handle and ABI version 1");
#if defined(_WIN32) || defined(__linux__)
        auto status = validate(*file->state);
        if (status != DDO_VMM_SUCCESS) return status;
        *info = file->state->info;
        return DDO_VMM_SUCCESS;
#else
        return fail(DDO_VMM_UNSUPPORTED, "native file source requires Windows or Linux");
#endif
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_file_read(
    const ddo_vmm_source_file* file, uint64_t offset, void* destination, size_t size
) {
    return boundary([&]() -> ddo_vmm_status {
        if (!file || !destination || size == 0)
            return fail(DDO_VMM_INVALID_ARGUMENT, "source read requires an open handle, buffer and positive size");
        const auto& state = *file->state;
        if (offset > state.info.size_bytes || size > state.info.size_bytes - offset)
            return fail(DDO_VMM_INVALID_ARGUMENT, "source read exceeds file bounds");
#if defined(_WIN32) || defined(__linux__)
        auto status = validate(state);
        if (status != DDO_VMM_SUCCESS) return status;
        // Same positioned reader as the native ring. This primitive does not
        // create another pool; asynchronous scheduling belongs to the transport.
        constexpr size_t chunk_bytes = 2 * 1024 * 1024;
        auto* bytes = static_cast<unsigned char*>(destination);
        std::string error;
        while (size) {
            const auto chunk = std::min(size, chunk_bytes);
            if (!ddo_source_io::read_file(state.handle, offset, bytes, chunk, error))
                return fail(DDO_VMM_IO_ERROR, error.c_str());
            bytes += chunk;
            offset += chunk;
            size -= chunk;
        }
        return validate(state);
#else
        return fail(DDO_VMM_UNSUPPORTED, "native file source requires Windows or Linux");
#endif
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_file_close(ddo_vmm_source_file** file) {
    return boundary([&]() -> ddo_vmm_status {
        if (!file) return fail(DDO_VMM_INVALID_ARGUMENT, "source handle address is required");
        delete *file;
        *file = nullptr;
        return DDO_VMM_SUCCESS;
    });
}
