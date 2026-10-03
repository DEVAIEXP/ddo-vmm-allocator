#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include "ddo_vmm.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// Shared positioned I/O primitives used by the existing ring and file sources.
// No CUDA context or worker-pool ownership is involved here.
namespace ddo_source_io {
HANDLE open_file(const char* path_utf8, std::string& error);
bool read_file(HANDLE file, uint64_t offset, void* destination, size_t size, std::string& error);
// Borrowed handle: source owner must outlive the asynchronous read.
HANDLE source_handle(const ddo_vmm_source_file* source);
ddo_vmm_status read_source_async(ddo_vmm_host_ring* ring, size_t slot,
    const ddo_vmm_source_file* source, uint64_t offset, size_t size);
// CPU completion observation only; does not consume the future or release
// ownership. Caller retains the ring/slot until wait_file_read retires it.
bool source_read_complete(ddo_vmm_host_ring* ring, size_t slot);
// Cache-owned rings start with metadata/workers only. Slot memory is committed
// on admission and released only after cache-owned DMA tickets have drained.
ddo_vmm_status create_lazy_ring(int device, size_t slot_bytes, size_t slots, ddo_vmm_host_ring** output);
ddo_vmm_status commit_slot(ddo_vmm_host_ring* ring, size_t slot);
// Persistent storage and CUDA registration have independent lifetimes.
ddo_vmm_status commit_storage(ddo_vmm_host_ring* ring, size_t slot);
ddo_vmm_status register_storage(ddo_vmm_host_ring* ring, size_t slot);
ddo_vmm_status unregister_storage(ddo_vmm_host_ring* ring, size_t slot);
ddo_vmm_status stage_host_bytes(ddo_vmm_host_ring* ring, size_t slot, const void* bytes, size_t size);
ddo_vmm_status release_slot(ddo_vmm_host_ring* ring, size_t slot);
}
#elif defined(__linux__)
#include "posix_file_io.h"
namespace ddo_source_io {
// Borrowed descriptor. The retained source owner must outlive each read.
file_handle source_handle(const ddo_vmm_source_file* source);
}
#endif
