#pragma once

// Native storage and admission only. No CUDA calls or allocator policy.
#include <cstddef>
#include <cstdint>
#include <string>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include "linux_host_memory.h"
#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#endif

namespace ddo_host_platform {
struct Headroom { uint64_t ram = 0, commit = 0; };
inline bool headroom(Headroom& result) {
#if defined(_WIN32)
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (!GlobalMemoryStatusEx(&memory)) return false;
    result = {memory.ullAvailPhys, memory.ullAvailPageFile};
    return true;
#elif defined(__linux__)
    return ddo_linux_memory::observe(result.ram, result.commit);
#else
    (void)result;
    return false;
#endif
}
inline bool permits(const Headroom& memory, uint64_t bytes, uint64_t reserve) {
    return bytes <= memory.ram && bytes <= memory.commit
        && reserve <= memory.ram - bytes && reserve <= memory.commit - bytes;
}
inline void* allocate(size_t size) {
#if defined(_WIN32)
    return VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#elif defined(__linux__)
    // No MAP_NORESERVE: keep normal kernel commit accounting. Population is
    // performed by source reads, not an eager full-model memset/mlock.
    void* address = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return address == MAP_FAILED ? nullptr : address;
#else
    (void)size;
    return nullptr;
#endif
}
inline bool release(void* address, size_t size) {
#if defined(_WIN32)
    (void)size;
    return VirtualFree(address, 0, MEM_RELEASE) != 0;
#elif defined(__linux__)
    return munmap(address, size) == 0;
#else
    (void)address; (void)size;
    return false;
#endif
}
}
