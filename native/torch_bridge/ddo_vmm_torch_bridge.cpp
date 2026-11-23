// Optional, Torch-version-specific registration of the native allocator.
//
// It performs the same registration as torch.cuda.memory.CUDAPluggableAllocator
// (createCustomAllocator with the already-loaded ddo_vmm_torch_malloc/free
// addresses) and adds only one check: a null result raises a Python exception
// before PyTorch can publish null tensor storage. Successful pointers and frees
// pass through unchanged. The bridge never loads the native library itself, so
// the pool configured by the installer is the one that allocates.
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/cuda/CUDAPluggableAllocator.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "checked_allocation.h"

namespace {
using Allocate = void* (*)(size_t, int, cudaStream_t);
using Release = void (*)(void*, size_t, int, cudaStream_t);
using LastRefusal = int (*)(std::uint64_t*, size_t);

std::mutex installation_mutex;
bool installed = false;

// The pool lock can be briefly held by a concurrent free; retry a bounded
// number of times without waiting on CUDA.
int read_last_refusal(LastRefusal last, std::uint64_t* report) {
    if (!last) return -1;
    int status = 2;
    for (int attempt = 0; attempt < 1000 && status == 2; ++attempt) {
        status = last(report, ddo_vmm_bridge::kReportFields);
        if (status == 2) std::this_thread::yield();
    }
    return status;
}

[[noreturn]] void raise_refusal(const ddo_vmm_bridge::RefusalClass& refusal) {
    if (refusal.capacity) TORCH_CHECK_WITH(OutOfMemoryError, false, refusal.message);
    TORCH_CHECK(false, refusal.message);
}

[[noreturn]] void reject(size_t bytes, int device, LastRefusal last) {
    std::uint64_t report[ddo_vmm_bridge::kReportFields]{};
    const int status = read_last_refusal(last, report);
    raise_refusal(ddo_vmm_bridge::classify_refusal(bytes, device, status, report));
}

void install(std::uintptr_t allocate_address, std::uintptr_t release_address, std::uintptr_t last_refusal_address) {
    std::scoped_lock lock(installation_mutex);
    TORCH_CHECK(!installed, "The DDO checked allocator is already installed; use a fresh process");
    TORCH_CHECK(allocate_address && release_address, "Native allocation exports are required");
    TORCH_CHECK(!c10::cuda::CUDACachingAllocator::get()->initialized(),
                "The CUDA allocator is already initialized; use the DDO fresh-child bootstrap");
    auto allocate = reinterpret_cast<Allocate>(allocate_address);
    auto release = reinterpret_cast<Release>(release_address);
    auto last = reinterpret_cast<LastRefusal>(last_refusal_address);
    auto allocator = torch::cuda::CUDAPluggableAllocator::createCustomAllocator(
        [allocate, last](size_t bytes, int device, cudaStream_t stream) -> void* {
            return ddo_vmm_bridge::checked_allocation(allocate, [=]() { reject(bytes, device, last); },
                                                      bytes, device, stream);
        },
        [release](void* pointer, size_t bytes, int device, cudaStream_t stream) {
            release(pointer, bytes, device, stream);
        });
    torch::cuda::CUDAPluggableAllocator::changeCurrentAllocator(allocator);
    installed = true;
}

// CPU probes: no allocator registration, CUDA call or native library use.
void probe_refusal(size_t bytes, int device, int report_status, std::vector<std::uint64_t> report) {
    TORCH_CHECK(report.empty() || report.size() == ddo_vmm_bridge::kReportFields, "Invalid probe report size");
    ddo_vmm_bridge::checked_allocation(
        []() -> void* { return nullptr; },
        [&]() { raise_refusal(ddo_vmm_bridge::classify_refusal(bytes, device, report_status,
                                                                report.empty() ? nullptr : report.data())); });
}

bool probe_success() {
    int marker = 0;
    return ddo_vmm_bridge::checked_allocation([&]() -> void* { return &marker; },
                                              []() { TORCH_CHECK(false, "unexpected rejection"); }) == &marker;
}
}  // namespace

PYBIND11_MODULE(ddo_vmm_torch_bridge, module) {
    module.attr("BRIDGE_ABI") = 1;
    module.def("install", torch::wrap_pybind_function(install));
    module.def("probe_refusal", torch::wrap_pybind_function(probe_refusal));
    module.def("probe_success", torch::wrap_pybind_function(probe_success));
}
