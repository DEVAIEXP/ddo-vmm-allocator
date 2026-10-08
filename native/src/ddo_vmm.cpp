#include "ddo_vmm.h"
#include "ddo_vmm_timeline.h"
#include "source_io.h"
#include "host_accounting.h"
#include "host_platform.h"
#include "ddo_vmm_read_wait.h"
#include "ddo_vmm_allocation_diagnostics.h"

#include <cuda.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#endif

#ifdef DDO_VMM_ALLOCATION_DIAGNOSTICS
namespace ddo_vmm_detail { AllocationDiagnostics allocation_diagnostics; }
// Wrap only this native implementation's calls; disabled builds call the driver directly.
#define cuMemCreate(...) ddo_vmm_detail::allocation_diagnostics.measure(ddo_vmm_detail::AllocationDiagnostics::create, [&] { return ::cuMemCreate(__VA_ARGS__); })
#define cuMemMap(...) ddo_vmm_detail::allocation_diagnostics.measure(ddo_vmm_detail::AllocationDiagnostics::map, [&] { return ::cuMemMap(__VA_ARGS__); })
#define cuMemSetAccess(...) ddo_vmm_detail::allocation_diagnostics.measure(ddo_vmm_detail::AllocationDiagnostics::access, [&] { const auto result = ::cuMemSetAccess(__VA_ARGS__); ddo_vmm_detail::allocation_diagnostics.access_result(result, CUDA_ERROR_NOT_READY, CUDA_ERROR_OUT_OF_MEMORY); return result; })
#define cuMemUnmap(...) ddo_vmm_detail::allocation_diagnostics.measure(ddo_vmm_detail::AllocationDiagnostics::unmap, [&] { return ::cuMemUnmap(__VA_ARGS__); })
#define cuMemRelease(...) ddo_vmm_detail::allocation_diagnostics.measure(ddo_vmm_detail::AllocationDiagnostics::release, [&] { return ::cuMemRelease(__VA_ARGS__); })
#define cuMemAddressReserve(...) ddo_vmm_detail::allocation_diagnostics.measure(ddo_vmm_detail::AllocationDiagnostics::reserve, [&] { return ::cuMemAddressReserve(__VA_ARGS__); })
#define cuMemAddressFree(...) ddo_vmm_detail::allocation_diagnostics.measure(ddo_vmm_detail::AllocationDiagnostics::address_free, [&] { return ::cuMemAddressFree(__VA_ARGS__); })
#endif

extern "C" int ddo_vmm_allocation_diagnostics_enable_v1(int enabled) {
#ifdef DDO_VMM_ALLOCATION_DIAGNOSTICS
    if (enabled != 0 && enabled != 1) return 1;
    if (enabled) return ddo_vmm_detail::allocation_diagnostics.start() ? 0 : 1;
    ddo_vmm_detail::allocation_diagnostics.enabled.store(false);
    return 0;
#else
    (void)enabled;
    return 2;
#endif
}
extern "C" int ddo_vmm_allocation_diagnostics_snapshot_v1(uint64_t* values, size_t count) {
#ifdef DDO_VMM_ALLOCATION_DIAGNOSTICS
    return ddo_vmm_detail::allocation_diagnostics.snapshot(values, count);
#else
    (void)values; (void)count;
    return 2;
#endif
}

extern "C" int ddo_vmm_allocation_diagnostics_snapshot_v2(uint64_t* values, size_t count) {
#ifdef DDO_VMM_ALLOCATION_DIAGNOSTICS
    return ddo_vmm_detail::allocation_diagnostics.snapshot_v2(values, count);
#else
    (void)values; (void)count;
    return 2;
#endif
}

namespace {

thread_local std::string last_error;

void set_error(const char* operation, CUresult result) {
    const char* error_name = "unknown";
    const char* error_message = "unknown CUDA driver error";
    cuGetErrorName(result, &error_name);
    cuGetErrorString(result, &error_message);
    last_error = std::string(operation) + " failed: " + error_name + " (" + error_message + ")";
}

bool checked(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) {
        return true;
    }
    set_error(operation, result);
    return false;
}

size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0 || value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return 0;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

bool device_properties(int ordinal, ddo_vmm_backing_kind backing_kind, CUdevice* device, CUmemAllocationProp* properties) {
    if (!checked(cuDeviceGet(device, ordinal), "cuDeviceGet")) {
        return false;
    }
    *properties = {};
    properties->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    if (backing_kind == DDO_VMM_BACKING_HOST) {
        properties->location.type = CU_MEM_LOCATION_TYPE_HOST;
        properties->location.id = 0;
    } else {
        properties->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        properties->location.id = ordinal;
    }
    properties->requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    return true;
}

bool ensure_context(int device_ordinal, CUcontext* retained_primary_context) {
    CUcontext current = nullptr;
    if (!checked(cuCtxGetCurrent(&current), "cuCtxGetCurrent")) {
        return false;
    }
    if (current != nullptr) {
        return true;
    }
    if (!checked(
            cuDevicePrimaryCtxRetain(retained_primary_context, device_ordinal),
            "cuDevicePrimaryCtxRetain"
        )) {
        return false;
    }
    if (!checked(cuCtxSetCurrent(*retained_primary_context), "cuCtxSetCurrent")) {
        cuDevicePrimaryCtxRelease(device_ordinal);
        *retained_primary_context = nullptr;
        return false;
    }
    return true;
}

#if defined(_WIN32)
bool utf8_path_to_wide(const char* path_utf8, std::vector<wchar_t>* out_path) {
    if (path_utf8 == nullptr || out_path == nullptr) {
        last_error = "native shard path arguments are invalid";
        return false;
    }
    const int required = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path_utf8, -1, nullptr, 0);
    if (required <= 0) {
        last_error = "native shard path is not valid UTF-8";
        return false;
    }
    out_path->resize(static_cast<size_t>(required));
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path_utf8, -1, out_path->data(), required) <= 0) {
        last_error = "native shard path conversion failed";
        return false;
    }
    return true;
}

HANDLE open_native_shard_file(const char* path_utf8) {
    std::vector<wchar_t> path;
    if (!utf8_path_to_wide(path_utf8, &path)) {
        return INVALID_HANDLE_VALUE;
    }
    HANDLE file = CreateFileW(
        path.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED, nullptr
    );
    if (file == INVALID_HANDLE_VALUE) {
        last_error = "native shard file open failed";
    }
    return file;
}

bool read_file_handle_into_buffer(HANDLE file, uint64_t offset, void* destination, size_t size) {
    if (file == INVALID_HANDLE_VALUE || destination == nullptr || size == 0 || size > std::numeric_limits<DWORD>::max()) {
        last_error = "native shard read arguments are invalid";
        return false;
    }
    OVERLAPPED request{};
    request.Offset = static_cast<DWORD>(offset & 0xffffffffULL);
    request.OffsetHigh = static_cast<DWORD>(offset >> 32);
    request.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (request.hEvent == nullptr) {
        last_error = "native shard read event creation failed (Win32 " + std::to_string(GetLastError()) + ")";
        return false;
    }
    DWORD read = 0;
    BOOL submitted = ReadFile(file, destination, static_cast<DWORD>(size), nullptr, &request);
    if (!submitted && GetLastError() != ERROR_IO_PENDING) {
        const DWORD error = GetLastError();
        CloseHandle(request.hEvent);
        last_error = "native shard positioned read submission failed (Win32 " + std::to_string(error) + ")";
        return false;
    }
    if (!GetOverlappedResult(file, &request, &read, TRUE) || read != size) {
        const DWORD error = GetLastError();
        CloseHandle(request.hEvent);
        last_error = "native shard positioned read failed or was short (Win32 " + std::to_string(error) + ")";
        return false;
    }
    CloseHandle(request.hEvent);
    return true;
}

bool read_file_into_buffer(const char* path_utf8, uint64_t offset, void* destination, size_t size) {
    if (path_utf8 == nullptr || destination == nullptr || size == 0 || size > std::numeric_limits<DWORD>::max()) {
        last_error = "native shard read arguments are invalid";
        return false;
    }
    HANDLE file = open_native_shard_file(path_utf8);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }
    const bool ok = read_file_handle_into_buffer(file, offset, destination, size);
    CloseHandle(file);
    return ok;
}
#endif

}  // namespace

#if defined(_WIN32)
namespace ddo_source_io {
HANDLE open_file(const char* path_utf8, std::string& error) {
    HANDLE file = open_native_shard_file(path_utf8);
    if (file == INVALID_HANDLE_VALUE) error = last_error;
    return file;
}

bool read_file(HANDLE file, uint64_t offset, void* destination, size_t size, std::string& error) {
    const bool succeeded = read_file_handle_into_buffer(file, offset, destination, size);
    if (!succeeded) error = last_error;
    return succeeded;
}
}
#endif

struct ddo_vmm_region {
    int device_ordinal = -1;
    CUdeviceptr address = 0;
    size_t requested_size = 0;
    size_t mapped_size = 0;
    size_t granularity = 0;
    CUmemGenericAllocationHandle allocation = 0;
    struct Mapping {
        size_t offset = 0;
        size_t size = 0;
        CUmemGenericAllocationHandle allocation = 0;
    };
    std::vector<Mapping> range_mappings;
    CUcontext retained_primary_context = nullptr;
    ddo_vmm_backing_kind backing_kind = DDO_VMM_BACKING_DEVICE;
};

struct ddo_vmm_model_arena {
    int device_ordinal = -1;
    std::mutex mutation_mutex;
    CUdeviceptr address = 0;
    size_t virtual_size = 0;
    size_t page_size = 0;
    size_t resident_budget = 0;
    size_t resident_pages = 0;
    uint64_t materialized_pages = 0;
    uint64_t reused_pages = 0;
    uint64_t evicted_pages = 0;
    uint64_t tail_evictions = 0;
    uint64_t pressure_evictions = 0;
    size_t fault_ceiling_page = 0;
    const char* admission_limit_reason = "resident capacity";
    CUevent compute_fence = nullptr;
    bool compute_fence_recorded = false;
    bool streaming_admission = false;
    CUmemGenericAllocationHandle view_bootstrap_allocation = 0;
    CUcontext retained_primary_context = nullptr;
    ddo_vmm_backing_kind backing_kind = DDO_VMM_BACKING_DEVICE;

    struct Page {
        CUmemGenericAllocationHandle allocation = 0;
        uint32_t pin_count = 0;
        uint64_t generation = 0;
        bool view_bootstrap_mapped = false;
    };
    std::vector<Page> pages;
};

struct ddo_vmm_execution_pool {
    struct Slot {
        ddo_vmm_model_arena* arena = nullptr;
        bool leased = false;
        bool materialized = false;
    };
    std::mutex mutex;
    std::vector<Slot> slots;
    size_t next = 0;
    ~ddo_vmm_execution_pool() {
        for (auto& slot : slots) ddo_vmm_model_arena_destroy(slot.arena);
    }
};

struct ddo_vmm_host_ring {
    int device_ordinal = -1;
    size_t slot_size = 0;
    bool accounted_staging = false;
    struct FileReadResult {
        bool succeeded = false;
        std::string error;
    };
    struct Slot {
        void* address = nullptr;
        CUevent completion = nullptr;
        bool in_flight = false;
        bool virtual_memory_registered = false;
#if defined(DDO_VMM_NATIVE_SOURCE_DISPATCHER)
        std::shared_future<FileReadResult> file_read;
#else
        std::future<FileReadResult> file_read;
#endif
        bool read_in_flight = false;
        size_t read_waiters = 0;
        bool read_busy() const { return read_in_flight || read_waiters != 0; }
    };
    std::mutex mutex;
    std::vector<Slot> slots;
    std::condition_variable read_waiters_retired;
    bool closing = false;
#if defined(_WIN32) || defined(__linux__)
    struct FileReadCompletion {
        std::promise<FileReadResult> promise;
        std::atomic<size_t> pending_chunks = 0;
        std::atomic<bool> failed = false;
        std::mutex error_mutex;
        std::string error;
    };
    struct FileReadTask {
        ddo_source_io::file_handle file = ddo_source_io::invalid_file_handle;
        uint64_t offset = 0;
        void* destination = nullptr;
        size_t size = 0;
        std::shared_ptr<FileReadCompletion> completion;
    };
    std::mutex source_mutex;
    std::condition_variable source_ready;
    std::condition_variable source_space;
    std::deque<FileReadTask> source_queue;
    std::vector<std::thread> source_workers;
    bool source_stopping = false;
    std::unordered_map<std::string, ddo_source_io::file_handle> shard_files;
    std::atomic<uint64_t> source_read_chunks = 0;
    std::atomic<uint64_t> source_read_bytes = 0;
    std::atomic<uint64_t> source_read_failures = 0;
#endif
};

#if defined(_WIN32) || defined(__linux__)
constexpr size_t kNativeSourceChunkBytes = 2 * 1024 * 1024;
constexpr size_t kNativeSourceWorkerLimit = 8;
constexpr size_t kNativeSourceQueueLimit = 256;

ddo_vmm_status enqueue_source_read_locked(
    ddo_vmm_host_ring* ring, ddo_vmm_host_ring::Slot& slot,
    ddo_source_io::file_handle file, uint64_t file_offset_bytes, size_t size_bytes
);

void complete_host_ring_source_read(
    const std::shared_ptr<ddo_vmm_host_ring::FileReadCompletion>& completion,
    bool succeeded,
    const std::string& error
) {
    if (!succeeded) {
        completion->failed.store(true, std::memory_order_release);
        std::lock_guard<std::mutex> lock(completion->error_mutex);
        if (completion->error.empty()) {
            completion->error = error.empty() ? "native shard source chunk failed" : error;
        }
    }
    if (completion->pending_chunks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::string completion_error;
        if (completion->failed.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lock(completion->error_mutex);
            completion_error = completion->error;
        }
        completion->promise.set_value({completion_error.empty(), std::move(completion_error)});
    }
}

bool enqueue_host_ring_source_task(ddo_vmm_host_ring* ring, ddo_vmm_host_ring::FileReadTask task) {
    {
        std::unique_lock<std::mutex> lock(ring->source_mutex);
        ring->source_space.wait(lock, [ring] {
            return ring->source_stopping || ring->source_queue.size() < kNativeSourceQueueLimit;
        });
        if (ring->source_stopping) {
            return false;
        }
        ring->source_queue.push_back(std::move(task));
    }
    ring->source_ready.notify_one();
    return true;
}

void host_ring_source_worker(ddo_vmm_host_ring* ring) {
    for (;;) {
        ddo_vmm_host_ring::FileReadTask task{};
        {
            std::unique_lock<std::mutex> lock(ring->source_mutex);
            ring->source_ready.wait(lock, [ring] {
                return ring->source_stopping || !ring->source_queue.empty();
            });
            if (ring->source_stopping && ring->source_queue.empty()) {
                return;
            }
            task = std::move(ring->source_queue.front());
            ring->source_queue.pop_front();
        }
        ring->source_space.notify_one();
        std::string error;
        const bool succeeded = ddo_source_io::read_file(task.file, task.offset, task.destination, task.size, error);
        ring->source_read_chunks.fetch_add(1, std::memory_order_relaxed);
        ring->source_read_bytes.fetch_add(task.size, std::memory_order_relaxed);
        if (!succeeded) {
            ring->source_read_failures.fetch_add(1, std::memory_order_relaxed);
        }
        complete_host_ring_source_read(task.completion, succeeded, error);
    }
}

bool start_host_ring_source_workers(ddo_vmm_host_ring* ring) {
    const size_t worker_count = std::min<size_t>(kNativeSourceWorkerLimit, ring->slots.size());
    try {
        for (size_t index = 0; index < worker_count; ++index) {
            ring->source_workers.emplace_back(host_ring_source_worker, ring);
        }
    } catch (const std::exception&) {
        {
            std::lock_guard<std::mutex> lock(ring->source_mutex);
            ring->source_stopping = true;
        }
        ring->source_ready.notify_all();
        for (std::thread& worker : ring->source_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        ring->source_workers.clear();
        last_error = "native shard source worker creation failed";
        return false;
    }
    return true;
}

void stop_host_ring_source_workers(ddo_vmm_host_ring* ring) {
    {
        std::lock_guard<std::mutex> lock(ring->source_mutex);
        ring->source_stopping = true;
    }
    ring->source_ready.notify_all();
    ring->source_space.notify_all();
    for (std::thread& worker : ring->source_workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    ring->source_workers.clear();
}
#endif

namespace {

struct PressureEntry {
    ddo_vmm_region* region = nullptr;
    bool pinned = false;
    uint64_t touch = 0;
};

struct ModelArenaPressureEntry {
    ddo_vmm_model_arena* arena = nullptr;
};

std::mutex pressure_mutex;
std::vector<PressureEntry> pressure_entries;
std::vector<ModelArenaPressureEntry> pressure_model_arenas;
uint64_t pressure_clock = 0;

struct DxgiBudgetGuardConfig {
    uint64_t local_reserve_bytes = 0;
    uint64_t shared_reserve_bytes = 0;
    uint64_t system_reserve_bytes = 0;
};

std::mutex dxgi_budget_guard_mutex;
std::vector<std::pair<int, DxgiBudgetGuardConfig>> dxgi_budget_guards;

PressureEntry* pressure_entry(ddo_vmm_region* region) {
    for (PressureEntry& entry : pressure_entries) {
        if (entry.region == region) {
            return &entry;
        }
    }
    return nullptr;
}

void register_model_arena_pressure(ddo_vmm_model_arena* arena) {
    if (arena == nullptr) {
        return;
    }
    std::scoped_lock lock(pressure_mutex);
    for (const ModelArenaPressureEntry& entry : pressure_model_arenas) {
        if (entry.arena == arena) {
            return;
        }
    }
    pressure_model_arenas.push_back({arena});
}

void unregister_model_arena_pressure(ddo_vmm_model_arena* arena) {
    std::scoped_lock lock(pressure_mutex);
    for (auto entry = pressure_model_arenas.begin(); entry != pressure_model_arenas.end(); ++entry) {
        if (entry->arena == arena) {
            pressure_model_arenas.erase(entry);
            return;
        }
    }
}

}  // namespace

#if defined(_WIN32)
bool query_windows_memory(int device_ordinal, ddo_vmm_windows_memory_info* out_info) {
    if (out_info == nullptr) {
        return false;
    }
    *out_info = {};
    CUdevice device = 0;
    char cuda_luid[8]{};
    unsigned int node_mask = 0;
    if (cuDeviceGet(&device, device_ordinal) != CUDA_SUCCESS
        || cuDeviceGetLuid(cuda_luid, &node_mask, device) != CUDA_SUCCESS) {
        return false;
    }
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory)))) {
        return false;
    }
    IDXGIAdapter1* adapter = nullptr;
    for (UINT index = 0; factory->EnumAdapters1(index, &adapter) != DXGI_ERROR_NOT_FOUND; ++index) {
        DXGI_ADAPTER_DESC1 description{};
        if (SUCCEEDED(adapter->GetDesc1(&description)) && memcmp(&description.AdapterLuid, cuda_luid, sizeof(cuda_luid)) == 0) {
            break;
        }
        adapter->Release();
        adapter = nullptr;
    }
    factory->Release();
    if (adapter == nullptr) {
        return false;
    }
    IDXGIAdapter3* adapter3 = nullptr;
    const HRESULT query_result = adapter->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3));
    adapter->Release();
    if (FAILED(query_result)) {
        return false;
    }
    DXGI_QUERY_VIDEO_MEMORY_INFO local{};
    DXGI_QUERY_VIDEO_MEMORY_INFO nonlocal{};
    const bool queried = SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local))
        && SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal));
    adapter3->Release();
    if (!queried) {
        return false;
    }
    MEMORYSTATUSEX system_info{};
    system_info.dwLength = sizeof(system_info);
    if (!GlobalMemoryStatusEx(&system_info)) {
        return false;
    }
    out_info->local_budget_bytes = local.Budget;
    out_info->local_usage_bytes = local.CurrentUsage;
    out_info->nonlocal_budget_bytes = nonlocal.Budget;
    out_info->nonlocal_usage_bytes = nonlocal.CurrentUsage;
    out_info->available_physical_bytes = system_info.ullAvailPhys;
    out_info->available_commit_bytes = system_info.ullAvailPageFile;
    return true;
}
#endif

bool dxgi_budget_guard_allows_materialize(
    int device_ordinal,
    size_t required_bytes,
    ddo_vmm_backing_kind backing_kind,
    bool allow_wddm_fallback = false
) {
    DxgiBudgetGuardConfig config{};
    {
        std::scoped_lock lock(dxgi_budget_guard_mutex);
        for (const auto& [ordinal, candidate] : dxgi_budget_guards) {
            if (ordinal == device_ordinal) {
                config = candidate;
                break;
            }
        }
    }
    if (
        config.local_reserve_bytes == 0 && config.shared_reserve_bytes == 0
        && config.system_reserve_bytes == 0
    ) {
        return true;
    }
#if defined(_WIN32)
    ddo_vmm_windows_memory_info info{};
    if (!query_windows_memory(device_ordinal, &info)) {
        return true;
    }
    const auto retained_budget = [](uint64_t budget, uint64_t reserve) {
        return budget > reserve ? budget - reserve : 0;
    };
    const uint64_t required = static_cast<uint64_t>(required_bytes);
    const uint64_t local_limit = retained_budget(info.local_budget_bytes, config.local_reserve_bytes);
    const uint64_t shared_limit = retained_budget(info.nonlocal_budget_bytes, config.shared_reserve_bytes);
    // DEVICE allocations are intentionally handed to WDDM as device
    // allocations.  The driver may place their physical pages in the
    // non-local/shared segment when local VRAM is exhausted, so the guard must
    // not reject them merely because the local budget is smaller than the
    // model.  HOST backing, when explicitly requested, is checked against the
    // shared budget directly.
    // The model-address arena explicitly delegates physical placement to
    // WDDM. Rejecting it at the local budget would prevent non-local backing.
    // Other VMM regions retain the conservative local-budget guard.
    const bool local_safe = allow_wddm_fallback || info.local_budget_bytes == 0
        || (info.local_usage_bytes <= local_limit && required <= local_limit - info.local_usage_bytes);
    const bool shared_safe = backing_kind != DDO_VMM_BACKING_HOST || info.nonlocal_budget_bytes == 0
        || (info.nonlocal_usage_bytes <= shared_limit && required <= shared_limit - info.nonlocal_usage_bytes);
    const bool system_safe = config.system_reserve_bytes == 0
        || (info.available_physical_bytes >= config.system_reserve_bytes + required
            && info.available_commit_bytes >= config.system_reserve_bytes + required);
    return local_safe && shared_safe && system_safe;
#else
    (void)device_ordinal;
    (void)required_bytes;
    (void)backing_kind;
    (void)allow_wddm_fallback;
    return true;
#endif
}

bool make_dxgi_budget_room(
    int device_ordinal,
    size_t required_bytes,
    ddo_vmm_backing_kind backing_kind,
    bool allow_wddm_fallback = false
) {
    while (!dxgi_budget_guard_allows_materialize(
        device_ordinal, required_bytes, backing_kind, allow_wddm_fallback
    )) {
        if (!ddo_vmm_pressure_evict_one(device_ordinal)) {
            return false;
        }
    }
    return true;
}

bool dxgi_budget_guard_configured(int device_ordinal) {
    std::scoped_lock lock(dxgi_budget_guard_mutex);
    for (const auto& [ordinal, config] : dxgi_budget_guards) {
        if (ordinal == device_ordinal) {
            return config.local_reserve_bytes != 0 || config.shared_reserve_bytes != 0
                || config.system_reserve_bytes != 0;
        }
    }
    return false;
}

namespace {

bool activate_region_context(const ddo_vmm_region* region, CUcontext* temporary_primary_context) {
    if (region->retained_primary_context != nullptr) {
        return checked(cuCtxSetCurrent(region->retained_primary_context), "cuCtxSetCurrent");
    }
    return ensure_context(region->device_ordinal, temporary_primary_context);
}

void release_temporary_primary_context(int device_ordinal, CUcontext temporary_primary_context) {
    if (temporary_primary_context != nullptr) {
        cuCtxSetCurrent(nullptr);
        cuDevicePrimaryCtxRelease(device_ordinal);
    }
}

bool activate_model_arena_context(const ddo_vmm_model_arena* arena, CUcontext* temporary_primary_context) {
    if (arena->retained_primary_context != nullptr) {
        return checked(cuCtxSetCurrent(arena->retained_primary_context), "cuCtxSetCurrent");
    }
    return ensure_context(arena->device_ordinal, temporary_primary_context);
}

size_t model_arena_page_begin(const ddo_vmm_model_arena* arena, size_t offset) {
    return offset / arena->page_size;
}

size_t model_arena_page_end(const ddo_vmm_model_arena* arena, size_t offset, size_t size) {
    return align_up(offset + size, arena->page_size) / arena->page_size;
}

uint64_t mix_signature(uint64_t value, uint64_t page, uint64_t generation) {
    value ^= (page + 0x9e3779b97f4a7c15ULL + (value << 6) + (value >> 2));
    value ^= (generation + 0xbf58476d1ce4e5b9ULL + (value << 7) + (value >> 3));
    return value == 0 ? 0xfeedfacecafebeefULL : value;
}

bool release_model_arena_page(ddo_vmm_model_arena* arena, size_t page_index) {
    if (arena == nullptr || page_index >= arena->pages.size()) {
        return false;
    }
    auto& page = arena->pages[page_index];
    if (page.allocation == 0) {
        return true;
    }
    arena->evicted_pages += 1;
    const CUdeviceptr page_address = arena->address + page_index * arena->page_size;
    if (!checked(cuMemUnmap(page_address, arena->page_size), "cuMemUnmap")) {
        return false;
    }
    if (!checked(cuMemRelease(page.allocation), "cuMemRelease")) {
        return false;
    }
    page.allocation = 0;
    page.pin_count = 0;
    if (arena->resident_pages > 0) {
        --arena->resident_pages;
    }
    return true;
}

bool model_arena_evict_one(
    ddo_vmm_model_arena* arena,
    size_t protected_begin = std::numeric_limits<size_t>::max(),
    size_t protected_end = std::numeric_limits<size_t>::max()
) {
    // Model weights are packed in observed execution order.  On pressure,
    // preserve the contiguous low-address prefix and recover the newest
    // eligible high-address page.  This is native arena policy, independent
    // of Python hook order or residency hints.
    for (size_t page_index = arena->pages.size(); page_index > 0; --page_index) {
        const size_t candidate_index = page_index - 1;
        const auto& page = arena->pages[candidate_index];
        if (page.allocation == 0 || page.pin_count != 0) {
            continue;
        }
        if (protected_begin != std::numeric_limits<size_t>::max()
            && protected_begin <= candidate_index && candidate_index < protected_end) {
            continue;
        }
        if (release_model_arena_page(arena, candidate_index)) {
            arena->tail_evictions += 1;
            return true;
        }
        return false;
    }
    return false;
}

void model_arena_limit_admission(
    ddo_vmm_model_arena* arena,
    size_t page_index,
    const char* reason
) {
    if (arena != nullptr) {
        arena->fault_ceiling_page = std::min(arena->fault_ceiling_page, page_index);
        arena->admission_limit_reason = reason;
    }
}

bool model_arena_can_admit_page(const ddo_vmm_model_arena* arena) {
    if (arena == nullptr || arena->resident_budget == 0) {
        return true;
    }
    const size_t budget_pages = std::max<size_t>(1, arena->resident_budget / arena->page_size);
    return arena->resident_pages < budget_pages;
}

size_t region_resident_size(const ddo_vmm_region* region) {
    if (region == nullptr) {
        return 0;
    }
    if (region->allocation != 0) {
        return region->mapped_size;
    }
    size_t total = 0;
    for (const auto& mapping : region->range_mappings) {
        total += mapping.size;
    }
    return total;
}

bool release_range_mappings(ddo_vmm_region* region, const char* operation) {
    for (auto& mapping : region->range_mappings) {
        if (mapping.allocation == 0) {
            continue;
        }
        if (!checked(cuMemUnmap(region->address + mapping.offset, mapping.size), operation)) {
            return false;
        }
        if (!checked(cuMemRelease(mapping.allocation), "cuMemRelease")) {
            return false;
        }
        mapping.allocation = 0;
    }
    region->range_mappings.clear();
    return true;
}

bool region_range_already_mapped(const ddo_vmm_region* region, size_t offset, size_t size) {
    if (region->allocation != 0) {
        return true;
    }
    const size_t end = offset + size;
    size_t cursor = offset;
    while (cursor < end) {
        bool advanced = false;
        for (const auto& mapping : region->range_mappings) {
            if (mapping.offset <= cursor && cursor < mapping.offset + mapping.size) {
                cursor = mapping.offset + mapping.size;
                advanced = true;
                break;
            }
        }
        if (!advanced) {
            return false;
        }
    }
    return true;
}

std::vector<std::pair<size_t, size_t>> missing_region_ranges(
    const ddo_vmm_region* region,
    size_t offset,
    size_t size
) {
    std::vector<std::pair<size_t, size_t>> missing;
    const size_t end = offset + size;
    size_t cursor = offset;
    while (cursor < end) {
        bool advanced = false;
        for (const auto& mapping : region->range_mappings) {
            if (mapping.offset <= cursor && cursor < mapping.offset + mapping.size) {
                cursor = mapping.offset + mapping.size;
                advanced = true;
                break;
            }
        }
        if (advanced) {
            continue;
        }
        size_t next = end;
        for (const auto& mapping : region->range_mappings) {
            if (mapping.offset > cursor && mapping.offset < next) {
                next = mapping.offset;
            }
        }
        missing.push_back({cursor, next - cursor});
        cursor = next;
    }
    return missing;
}

ddo_vmm_status materialize_region_range(
    ddo_vmm_region* region,
    size_t aligned_offset,
    size_t aligned_size
) {
    if (region == nullptr || aligned_size == 0 || aligned_offset > region->mapped_size
        || aligned_size > region->mapped_size - aligned_offset) {
        last_error = "VMM region range must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    if (region_range_already_mapped(region, aligned_offset, aligned_size)) {
        return DDO_VMM_SUCCESS;
    }
    if (region->allocation != 0) {
        return DDO_VMM_SUCCESS;
    }
    const auto missing_ranges = missing_region_ranges(region, aligned_offset, aligned_size);

    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(region->device_ordinal, region->backing_kind, &device, &properties)) {
        return DDO_VMM_CUDA_ERROR;
    }
    const bool guard_configured = dxgi_budget_guard_configured(region->device_ordinal);
    std::vector<ddo_vmm_region::Mapping> added_mappings;
    for (const auto& [range_offset, range_size] : missing_ranges) {
        if (!make_dxgi_budget_room(region->device_ordinal, range_size, region->backing_kind)) {
            set_error("dxgi_budget_guard", CUDA_ERROR_OUT_OF_MEMORY);
            return DDO_VMM_OUT_OF_MEMORY;
        }

        CUmemGenericAllocationHandle allocation = 0;
        while (cuMemCreate(&allocation, range_size, &properties, 0) != CUDA_SUCCESS) {
            allocation = 0;
            if (guard_configured) {
                set_error("cuMemCreate", CUDA_ERROR_OUT_OF_MEMORY);
                return DDO_VMM_OUT_OF_MEMORY;
            }
            if (!ddo_vmm_pressure_evict_one(region->device_ordinal)) {
                set_error("cuMemCreate", CUDA_ERROR_OUT_OF_MEMORY);
                return DDO_VMM_OUT_OF_MEMORY;
            }
        }
        if (allocation == 0) {
            return DDO_VMM_OUT_OF_MEMORY;
        }
        const CUdeviceptr target = region->address + range_offset;
        if (!checked(cuMemMap(target, range_size, 0, allocation, 0), "cuMemMap")) {
            cuMemRelease(allocation);
            return DDO_VMM_CUDA_ERROR;
        }
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = region->device_ordinal;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (!checked(cuMemSetAccess(target, range_size, &access, 1), "cuMemSetAccess")) {
            cuMemUnmap(target, range_size);
            cuMemRelease(allocation);
            return DDO_VMM_CUDA_ERROR;
        }
        added_mappings.push_back({range_offset, range_size, allocation});
    }
    region->range_mappings.insert(region->range_mappings.end(), added_mappings.begin(), added_mappings.end());
    return DDO_VMM_SUCCESS;
}

}  // namespace

extern "C" ddo_vmm_status ddo_vmm_region_create(
    int device_ordinal,
    size_t size_bytes,
    ddo_vmm_region** out_region
) {
    return ddo_vmm_region_create_ex(device_ordinal, size_bytes, DDO_VMM_BACKING_DEVICE, out_region);
}

extern "C" ddo_vmm_status ddo_vmm_region_create_ex(
    int device_ordinal,
    size_t size_bytes,
    ddo_vmm_backing_kind backing_kind,
    ddo_vmm_region** out_region
) {
    if (out_region == nullptr || size_bytes == 0 || device_ordinal < 0) {
        last_error = "device ordinal, size, and output pointer must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    *out_region = nullptr;
    last_error.clear();

    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    auto region = std::make_unique<ddo_vmm_region>();
    region->device_ordinal = device_ordinal;
    region->backing_kind = backing_kind;
    region->requested_size = size_bytes;
    if (!ensure_context(device_ordinal, &region->retained_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }

    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(device_ordinal, backing_kind, &device, &properties)) {
        return DDO_VMM_CUDA_ERROR;
    }

    size_t granularity = 0;
    if (!checked(
            cuMemGetAllocationGranularity(
                &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM
            ),
            "cuMemGetAllocationGranularity"
        )) {
        return DDO_VMM_CUDA_ERROR;
    }
    const size_t mapped_size = align_up(size_bytes, granularity);
    if (mapped_size == 0) {
        last_error = "requested size overflows CUDA VMM granularity alignment";
        return DDO_VMM_INVALID_ARGUMENT;
    }

    region->mapped_size = mapped_size;
    region->granularity = granularity;

    if (!checked(
            cuMemAddressReserve(&region->address, mapped_size, granularity, 0, 0),
            "cuMemAddressReserve"
        )) {
        if (region->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_CUDA_ERROR;
    }
    if (!checked(cuMemCreate(&region->allocation, mapped_size, &properties, 0), "cuMemCreate")) {
        cuMemAddressFree(region->address, region->mapped_size);
        if (region->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_OUT_OF_MEMORY;
    }
    if (!checked(cuMemMap(region->address, mapped_size, 0, region->allocation, 0), "cuMemMap")) {
        cuMemRelease(region->allocation);
        cuMemAddressFree(region->address, region->mapped_size);
        if (region->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_CUDA_ERROR;
    }

    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_ordinal;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!checked(cuMemSetAccess(region->address, mapped_size, &access, 1), "cuMemSetAccess")) {
        cuMemUnmap(region->address, region->mapped_size);
        cuMemRelease(region->allocation);
        cuMemAddressFree(region->address, region->mapped_size);
        if (region->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_CUDA_ERROR;
    }

    *out_region = region.release();
    return DDO_VMM_SUCCESS;
}

extern "C" void ddo_vmm_region_destroy(ddo_vmm_region* region) {
    if (region == nullptr) {
        return;
    }
    ddo_vmm_pressure_unregister(region);
    if (cuInit(0) == CUDA_SUCCESS) {
        CUcontext temporary_primary_context = nullptr;
        if (activate_region_context(region, &temporary_primary_context)) {
            if (region->allocation != 0 && region->address != 0 && region->mapped_size != 0) {
                cuMemUnmap(region->address, region->mapped_size);
            }
            if (region->allocation != 0) {
                cuMemRelease(region->allocation);
            }
            release_range_mappings(region, "cuMemUnmap");
            if (region->address != 0 && region->mapped_size != 0) {
                cuMemAddressFree(region->address, region->mapped_size);
            }
        }
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        if (region->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(region->device_ordinal);
        }
    }
    delete region;
}

extern "C" ddo_vmm_status ddo_vmm_region_evict(ddo_vmm_region* region) {
    if (region == nullptr) {
        last_error = "VMM region must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    last_error.clear();
    if (region->allocation == 0 && region->range_mappings.empty()) {
        return DDO_VMM_SUCCESS;
    }
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_region_context(region, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    if (region->allocation != 0) {
        if (!checked(cuMemUnmap(region->address, region->mapped_size), "cuMemUnmap")) {
            release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        if (!checked(cuMemRelease(region->allocation), "cuMemRelease")) {
            release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        region->allocation = 0;
    }
    if (!release_range_mappings(region, "cuMemUnmap")) {
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_region_materialize(ddo_vmm_region* region) {
    if (region == nullptr) {
        last_error = "VMM region must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    last_error.clear();
    if (region->allocation != 0) {
        return DDO_VMM_SUCCESS;
    }
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_region_context(region, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    if (!region->range_mappings.empty() && !release_range_mappings(region, "cuMemUnmap")) {
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(region->device_ordinal, region->backing_kind, &device, &properties)) {
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    CUmemGenericAllocationHandle allocation = 0;
    const bool guard_configured = dxgi_budget_guard_configured(region->device_ordinal);
    if (!make_dxgi_budget_room(region->device_ordinal, region->mapped_size, region->backing_kind)) {
        set_error("dxgi_budget_guard", CUDA_ERROR_OUT_OF_MEMORY);
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_OUT_OF_MEMORY;
    }
    while (cuMemCreate(&allocation, region->mapped_size, &properties, 0) != CUDA_SUCCESS) {
        allocation = 0;
        if (guard_configured) {
            set_error("cuMemCreate", CUDA_ERROR_OUT_OF_MEMORY);
            release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
            return DDO_VMM_OUT_OF_MEMORY;
        }
        if (!ddo_vmm_pressure_evict_one(region->device_ordinal)) {
            set_error("cuMemCreate", CUDA_ERROR_OUT_OF_MEMORY);
            release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
            return DDO_VMM_OUT_OF_MEMORY;
        }
    }
    if (allocation == 0) {
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_OUT_OF_MEMORY;
    }
    if (!checked(cuMemMap(region->address, region->mapped_size, 0, allocation, 0), "cuMemMap")) {
        cuMemRelease(allocation);
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = region->device_ordinal;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (!checked(cuMemSetAccess(region->address, region->mapped_size, &access, 1), "cuMemSetAccess")) {
        cuMemUnmap(region->address, region->mapped_size);
        cuMemRelease(allocation);
        release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    region->allocation = allocation;
    release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_region_materialize_range(
    ddo_vmm_region* region,
    size_t offset_bytes,
    size_t size_bytes
) {
    if (region == nullptr || size_bytes == 0) {
        last_error = "VMM region range must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    last_error.clear();
    if (offset_bytes > region->mapped_size || size_bytes > region->mapped_size - offset_bytes) {
        last_error = "VMM region range is outside the reserved address span";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t granularity = region->granularity == 0 ? 1 : region->granularity;
    const size_t aligned_offset = (offset_bytes / granularity) * granularity;
    const size_t range_end = align_up(offset_bytes + size_bytes, granularity);
    if (range_end == 0 || range_end > region->mapped_size || range_end < aligned_offset) {
        last_error = "VMM range overflows CUDA granularity alignment";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t aligned_size = range_end - aligned_offset;
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_region_context(region, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    ddo_vmm_status status = materialize_region_range(region, aligned_offset, aligned_size);
    release_temporary_primary_context(region->device_ordinal, temporary_primary_context);
    return status;
}

extern "C" size_t ddo_vmm_region_resident_size(const ddo_vmm_region* region) {
    return region_resident_size(region);
}

extern "C" uintptr_t ddo_vmm_region_address(const ddo_vmm_region* region) {
    return region == nullptr ? 0 : static_cast<uintptr_t>(region->address);
}

extern "C" size_t ddo_vmm_region_requested_size(const ddo_vmm_region* region) {
    return region == nullptr ? 0 : region->requested_size;
}

extern "C" size_t ddo_vmm_region_mapped_size(const ddo_vmm_region* region) {
    return region == nullptr ? 0 : region->mapped_size;
}

extern "C" int ddo_vmm_region_is_materialized(const ddo_vmm_region* region) {
    return region != nullptr && region_resident_size(region) == region->mapped_size ? 1 : 0;
}

extern "C" size_t ddo_vmm_device_granularity(int device_ordinal) {
    last_error.clear();
    if (device_ordinal < 0 || !checked(cuInit(0), "cuInit")) {
        return 0;
    }
    CUcontext retained_primary_context = nullptr;
    if (!ensure_context(device_ordinal, &retained_primary_context)) {
        return 0;
    }
    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(device_ordinal, DDO_VMM_BACKING_DEVICE, &device, &properties)) {
        return 0;
    }
    size_t granularity = 0;
    const size_t result = checked(
               cuMemGetAllocationGranularity(
                   &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM
               ),
               "cuMemGetAllocationGranularity"
           )
        ? granularity
        : 0;
    if (retained_primary_context != nullptr) {
        cuCtxSetCurrent(nullptr);
        cuDevicePrimaryCtxRelease(device_ordinal);
    }
    return result;
}

extern "C" ddo_vmm_status ddo_vmm_model_arena_create(
    int device_ordinal,
    size_t virtual_size_bytes,
    size_t resident_budget_bytes,
    size_t page_size_bytes,
    ddo_vmm_backing_kind backing_kind,
    ddo_vmm_model_arena** out_arena
) {
    if (out_arena == nullptr || virtual_size_bytes == 0 || device_ordinal < 0) {
        last_error = "device ordinal, virtual size, and output pointer must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    *out_arena = nullptr;
    last_error.clear();
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }

    auto arena = std::make_unique<ddo_vmm_model_arena>();
    arena->device_ordinal = device_ordinal;
    arena->resident_budget = resident_budget_bytes;
    arena->backing_kind = backing_kind;
    if (!ensure_context(device_ordinal, &arena->retained_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }

    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(device_ordinal, backing_kind, &device, &properties)) {
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_CUDA_ERROR;
    }

    size_t granularity = 0;
    if (!checked(
            cuMemGetAllocationGranularity(
                &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_MINIMUM
            ),
            "cuMemGetAllocationGranularity"
        )) {
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_CUDA_ERROR;
    }
    const size_t requested_page_size = page_size_bytes == 0 ? granularity : page_size_bytes;
    const size_t page_size = align_up(requested_page_size, granularity);
    if (page_size == 0) {
        last_error = "model arena page size overflows CUDA VMM granularity alignment";
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t virtual_size = align_up(virtual_size_bytes, page_size);
    if (virtual_size == 0) {
        last_error = "model arena virtual size overflows page alignment";
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t page_count = virtual_size / page_size;
    if (page_count == 0 || page_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        last_error = "model arena page count is invalid";
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_INVALID_ARGUMENT;
    }

    if (!checked(cuMemAddressReserve(&arena->address, virtual_size, page_size, 0, 0), "cuMemAddressReserve")) {
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device_ordinal);
        }
        return DDO_VMM_CUDA_ERROR;
    }
    arena->virtual_size = virtual_size;
    arena->page_size = page_size;
    arena->fault_ceiling_page = page_count;
    arena->pages.resize(page_count);

    *out_arena = arena.release();
    register_model_arena_pressure(*out_arena);
    return DDO_VMM_SUCCESS;
}

extern "C" void ddo_vmm_model_arena_destroy(ddo_vmm_model_arena* arena) {
    if (arena == nullptr) {
        return;
    }
    // Make the address unreachable to a concurrent allocator-pressure pass
    // before releasing its virtual reservation or physical backing.
    unregister_model_arena_pressure(arena);
    std::unique_lock mutation_lock(arena->mutation_mutex);
    if (cuInit(0) == CUDA_SUCCESS) {
        CUcontext temporary_primary_context = nullptr;
        if (activate_model_arena_context(arena, &temporary_primary_context)) {
            // Execution-buffer pages remain pinned, including during compute.
            // Never release them before their native completion fence drains.
            if (arena->compute_fence_recorded
                && cuEventSynchronize(arena->compute_fence) != CUDA_SUCCESS) {
                release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
                return; // Retain backing on failure rather than free live data.
            }
            if (arena->compute_fence != nullptr) {
                cuEventDestroy(arena->compute_fence);
                arena->compute_fence = nullptr;
            }
            for (size_t page_index = 0; page_index < arena->pages.size(); ++page_index) {
                auto& page = arena->pages[page_index];
                if (page.view_bootstrap_mapped) {
                    cuMemUnmap(arena->address + page_index * arena->page_size, arena->page_size);
                    page.view_bootstrap_mapped = false;
                }
                release_model_arena_page(arena, page_index);
            }
            if (arena->view_bootstrap_allocation != 0) {
                cuMemRelease(arena->view_bootstrap_allocation);
                arena->view_bootstrap_allocation = 0;
            }
            if (arena->address != 0 && arena->virtual_size != 0) {
                cuMemAddressFree(arena->address, arena->virtual_size);
            }
        }
        release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
        if (arena->retained_primary_context != nullptr) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(arena->device_ordinal);
        }
    }
    mutation_lock.unlock();
    delete arena;
}

extern "C" ddo_vmm_status ddo_vmm_model_arena_prepare_view_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes
) {
    std::unique_lock<std::mutex> mutation_lock;
    if (arena != nullptr) {
        mutation_lock = std::unique_lock<std::mutex>(arena->mutation_mutex);
    }
    if (arena == nullptr || size_bytes == 0 || offset_bytes > arena->virtual_size
        || size_bytes > arena->virtual_size - offset_bytes) {
        last_error = "model arena view range is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t begin_page = model_arena_page_begin(arena, offset_bytes);
    const size_t end_page = model_arena_page_end(arena, offset_bytes, size_bytes);
    if (begin_page >= end_page || end_page > arena->pages.size()) {
        last_error = "model arena view page range is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(arena->device_ordinal, arena->backing_kind, &device, &properties)) {
        release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    if (arena->view_bootstrap_allocation == 0
        && !checked(cuMemCreate(&arena->view_bootstrap_allocation, arena->page_size, &properties, 0), "cuMemCreate")) {
        release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    std::vector<size_t> mapped_pages;
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = arena->device_ordinal;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    for (size_t page_index = begin_page; page_index < end_page; ++page_index) {
        auto& page = arena->pages[page_index];
        if (page.allocation != 0 || page.view_bootstrap_mapped) {
            continue;
        }
        const CUdeviceptr page_address = arena->address + page_index * arena->page_size;
        if (cuMemMap(page_address, arena->page_size, 0, arena->view_bootstrap_allocation, 0) != CUDA_SUCCESS
            || cuMemSetAccess(page_address, arena->page_size, &access, 1) != CUDA_SUCCESS) {
            for (const size_t mapped_index : mapped_pages) {
                cuMemUnmap(arena->address + mapped_index * arena->page_size, arena->page_size);
                arena->pages[mapped_index].view_bootstrap_mapped = false;
            }
            last_error = "unable to map transient model arena view backing";
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        page.view_bootstrap_mapped = true;
        mapped_pages.push_back(page_index);
    }
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_model_arena_release_view_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes
) {
    std::unique_lock<std::mutex> mutation_lock;
    if (arena != nullptr) {
        mutation_lock = std::unique_lock<std::mutex>(arena->mutation_mutex);
    }
    if (arena == nullptr || size_bytes == 0 || offset_bytes > arena->virtual_size
        || size_bytes > arena->virtual_size - offset_bytes) {
        last_error = "model arena view range is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t begin_page = model_arena_page_begin(arena, offset_bytes);
    const size_t end_page = model_arena_page_end(arena, offset_bytes, size_bytes);
    if (begin_page >= end_page || end_page > arena->pages.size()) {
        last_error = "model arena view page range is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    for (size_t page_index = begin_page; page_index < end_page; ++page_index) {
        auto& page = arena->pages[page_index];
        if (!page.view_bootstrap_mapped) {
            continue;
        }
        if (!checked(cuMemUnmap(arena->address + page_index * arena->page_size, arena->page_size), "cuMemUnmap")) {
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        page.view_bootstrap_mapped = false;
    }
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_model_arena_fault_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes,
    uint64_t* out_signature
) {
#ifdef DDO_VMM_ALLOCATION_DIAGNOSTICS
    ddo_vmm_detail::AllocationDiagnostics::FaultScope fault_diagnostic(ddo_vmm_detail::allocation_diagnostics);
#endif
    std::unique_lock<std::mutex> mutation_lock;
    if (arena != nullptr) {
        mutation_lock = std::unique_lock<std::mutex>(arena->mutation_mutex);
    }
    if (arena == nullptr || size_bytes == 0 || out_signature == nullptr) {
        last_error = "model arena range and signature output must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    last_error.clear();
    if (offset_bytes > arena->virtual_size || size_bytes > arena->virtual_size - offset_bytes) {
        last_error = "model arena range is outside the reserved address span";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t begin_page = model_arena_page_begin(arena, offset_bytes);
    const size_t end_page = model_arena_page_end(arena, offset_bytes, size_bytes);
    if (begin_page >= end_page || end_page > arena->pages.size()) {
        last_error = "model arena page range is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    if (begin_page >= arena->fault_ceiling_page
        || (arena->streaming_admission && end_page > arena->fault_ceiling_page)) {
        last_error = std::string("model arena admission boundary reached by ")
            + arena->admission_limit_reason;
        return DDO_VMM_OUT_OF_MEMORY;
    }
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }

    CUdevice device = 0;
    CUmemAllocationProp properties{};
    if (!device_properties(arena->device_ordinal, arena->backing_kind, &device, &properties)) {
        release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
    for (size_t page_index = begin_page; page_index < end_page; ++page_index) {
        auto& page = arena->pages[page_index];
        if (page.allocation != 0) {
            arena->reused_pages += 1;
            continue;
        }
        if (page.view_bootstrap_mapped) {
            if (!checked(cuMemUnmap(arena->address + page_index * arena->page_size, arena->page_size), "cuMemUnmap")) {
                release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
                return DDO_VMM_CUDA_ERROR;
            }
            page.view_bootstrap_mapped = false;
        }
        if (page_index >= arena->fault_ceiling_page) {
            last_error = std::string("model arena admission boundary reached by ")
                + arena->admission_limit_reason;
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_OUT_OF_MEMORY;
        }
        // A capacity budget limits physical residency, not the virtual model
        // address span. Once an execution event has released an older pin,
        // reuse its least-recently-used page for the current range. If every
        // page is still in flight, report a recoverable capacity condition so
        // the adapter can retire precisely one completed CUDA event.
        while (!model_arena_can_admit_page(arena)) {
            if (!model_arena_evict_one(arena, begin_page, end_page)) {
                last_error = "model arena admission boundary reached by resident capacity";
                release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
                return DDO_VMM_OUT_OF_MEMORY;
            }
        }
        if (!make_dxgi_budget_room(
            arena->device_ordinal, arena->page_size, arena->backing_kind, true
        )) {
            model_arena_limit_admission(arena, page_index, "memory guard");
            last_error = "model arena admission boundary reached by memory guard";
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_OUT_OF_MEMORY;
        }
        CUmemGenericAllocationHandle allocation = 0;
        if (cuMemCreate(&allocation, arena->page_size, &properties, 0) != CUDA_SUCCESS) {
            model_arena_limit_admission(arena, page_index, "CUDA allocation pressure");
            last_error = "model arena admission boundary reached by CUDA allocation pressure";
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_OUT_OF_MEMORY;
        }
        const CUdeviceptr page_address = arena->address + page_index * arena->page_size;
        if (!checked(cuMemMap(page_address, arena->page_size, 0, allocation, 0), "cuMemMap")) {
            cuMemRelease(allocation);
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        CUmemAccessDesc access{};
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = arena->device_ordinal;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUresult access_result = CUDA_ERROR_NOT_READY;
        for (int attempt = 0; attempt < 4 && access_result == CUDA_ERROR_NOT_READY; ++attempt) {
            access_result = cuMemSetAccess(page_address, arena->page_size, &access, 1);
            if (access_result == CUDA_ERROR_NOT_READY) {
#ifdef DDO_VMM_ALLOCATION_DIAGNOSTICS
                ddo_vmm_detail::allocation_diagnostics.measure(
                    ddo_vmm_detail::AllocationDiagnostics::access_retry_sync,
                    [&] { return cuCtxSynchronize(); });
#else
                cuCtxSynchronize();
#endif
            }
        }
        if (!checked(access_result, "cuMemSetAccess")) {
            cuMemUnmap(page_address, arena->page_size);
            cuMemRelease(allocation);
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        page.allocation = allocation;
        page.generation += 1;
        arena->resident_pages += 1;
        arena->materialized_pages += 1;
    }

    uint64_t signature = 0x1234abcd5678ef00ULL;
    for (size_t page_index = begin_page; page_index < end_page; ++page_index) {
        auto& page = arena->pages[page_index];
        page.pin_count += 1;
        signature = mix_signature(signature, static_cast<uint64_t>(page_index), page.generation);
    }
    *out_signature = signature;
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_model_arena_peek_range_signature(
    ddo_vmm_model_arena* arena, size_t offset_bytes, size_t size_bytes,
    uint64_t* out_signature, int* out_resident
) {
    if (!arena || !out_signature || !out_resident || !size_bytes) {
        last_error = "model arena observation requires a valid range and outputs";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(arena->mutation_mutex);
    if (offset_bytes > arena->virtual_size || size_bytes > arena->virtual_size - offset_bytes) {
        last_error = "model arena observation is outside the reserved address span";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    *out_signature = 0;
    *out_resident = 0;
    const size_t begin = model_arena_page_begin(arena, offset_bytes);
    const size_t end = model_arena_page_end(arena, offset_bytes, size_bytes);
    uint64_t signature = 0x1234abcd5678ef00ULL;
    for (size_t index = begin; index < end; ++index) {
        const auto& page = arena->pages[index];
        if (!page.allocation) return DDO_VMM_SUCCESS;
        signature = mix_signature(signature, static_cast<uint64_t>(index), page.generation);
    }
    *out_signature = signature;
    *out_resident = 1;
    last_error.clear();
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_model_arena_copy_from_host(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes,
    const void* host_ptr
) {
    if (arena == nullptr || host_ptr == nullptr || size_bytes == 0 ||
        offset_bytes > arena->virtual_size || size_bytes > arena->virtual_size - offset_bytes) {
        return DDO_VMM_INVALID_ARGUMENT;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUresult result = cuMemcpyHtoD(arena->address + offset_bytes, host_ptr, size_bytes);
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    if (result != CUDA_SUCCESS) {
        set_error("cuMemcpyHtoD", result);
        return DDO_VMM_CUDA_ERROR;
    }
    return DDO_VMM_SUCCESS;
}
extern "C" ddo_vmm_status ddo_vmm_model_arena_copy_from_host_async(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes,
    const void* host_ptr,
    void* cuda_stream
) {
    if (arena == nullptr || host_ptr == nullptr || size_bytes == 0 ||
        offset_bytes > arena->virtual_size || size_bytes > arena->virtual_size - offset_bytes) {
        return DDO_VMM_INVALID_ARGUMENT;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUresult result = cuMemcpyHtoDAsync(
        arena->address + offset_bytes,
        host_ptr,
        size_bytes,
        reinterpret_cast<CUstream>(cuda_stream)
    );
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    if (result != CUDA_SUCCESS) {
        set_error("cuMemcpyHtoDAsync", result);
        return DDO_VMM_CUDA_ERROR;
    }
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_host_ring_copy_to_arena_async(
    ddo_vmm_host_ring* ring,
    size_t slot_index,
    size_t slot_offset_bytes,
    ddo_vmm_model_arena* arena,
    size_t arena_offset_bytes,
    size_t size_bytes,
    void* cuda_stream
) {
    if (ring == nullptr || arena == nullptr || slot_index >= ring->slots.size() || size_bytes == 0 ||
        slot_offset_bytes > ring->slot_size || size_bytes > ring->slot_size - slot_offset_bytes ||
        arena_offset_bytes > arena->virtual_size || size_bytes > arena->virtual_size - arena_offset_bytes) {
        last_error = "host ring copy arguments are outside their declared ranges";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    if (ring->device_ordinal != arena->device_ordinal) {
        last_error = "host ring and model arena must use the same CUDA device";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(ring->mutex);
    auto& slot = ring->slots[slot_index];
    if (slot.read_busy()) {
        last_error = "host ring source read must finish before DMA";
        return DDO_VMM_OUT_OF_MEMORY;
    }
    if (slot.in_flight && cuEventQuery(slot.completion) == CUDA_ERROR_NOT_READY) {
        last_error = "host ring slot is still in flight";
        return DDO_VMM_OUT_OF_MEMORY;
    }
    slot.in_flight = false;
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    const auto* source = static_cast<const unsigned char*>(slot.address) + slot_offset_bytes;
    CUresult result = cuMemcpyHtoDAsync(
        arena->address + arena_offset_bytes,
        source,
        size_bytes,
        reinterpret_cast<CUstream>(cuda_stream)
    );
    if (result != CUDA_SUCCESS) {
        release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
        set_error("cuMemcpyHtoDAsync(host ring)", result);
        return DDO_VMM_CUDA_ERROR;
    }
    result = cuEventRecord(slot.completion, reinterpret_cast<CUstream>(cuda_stream));
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    if (result != CUDA_SUCCESS) {
        set_error("cuEventRecord(host ring)", result);
        return DDO_VMM_CUDA_ERROR;
    }
    slot.in_flight = true;
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_host_ring_copy_ticket_to_arena_async(
    ddo_vmm_host_ring* ring,
    const size_t* slot_indices,
    ddo_vmm_model_arena* arena,
    const size_t* arena_offsets_bytes,
    const size_t* sizes_bytes,
    size_t range_count,
    void* cuda_stream
) {
    if (ring == nullptr || arena == nullptr || slot_indices == nullptr || arena_offsets_bytes == nullptr
        || sizes_bytes == nullptr || range_count == 0 || ring->device_ordinal != arena->device_ordinal) {
        last_error = "host ring ticket copy arguments are invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(ring->mutex);
    for (size_t index = 0; index < range_count; ++index) {
        const size_t slot_index = slot_indices[index];
        if (slot_index >= ring->slots.size() || sizes_bytes[index] == 0 || sizes_bytes[index] > ring->slot_size
            || arena_offsets_bytes[index] > arena->virtual_size
            || sizes_bytes[index] > arena->virtual_size - arena_offsets_bytes[index]) {
            last_error = "host ring ticket range is outside its declared bounds";
            return DDO_VMM_INVALID_ARGUMENT;
        }
        auto& slot = ring->slots[slot_index];
        if (slot.in_flight && cuEventQuery(slot.completion) == CUDA_ERROR_NOT_READY) {
            last_error = "host ring ticket slot is still in flight";
            return DDO_VMM_OUT_OF_MEMORY;
        }
        slot.in_flight = false;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    for (size_t index = 0; index < range_count; ++index) {
        auto& slot = ring->slots[slot_indices[index]];
        // Do not form a source-ticket barrier here. As each registered slot
        // becomes readable, immediately enqueue its H2D copy while later
        // source windows continue on the native worker pool.
        if (slot.read_in_flight) {
            ddo_vmm_host_ring::FileReadResult read_result{};
            try {
                read_result = slot.file_read.get();
            } catch (const std::exception&) {
                slot.read_in_flight = false;
                release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
                last_error = "native shard source worker raised an exception";
                return DDO_VMM_CUDA_ERROR;
            }
            slot.read_in_flight = false;
            if (!read_result.succeeded) {
                release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
                last_error = read_result.error.empty() ? "native shard source worker failed" : read_result.error;
                return DDO_VMM_CUDA_ERROR;
            }
        }
        CUresult result = cuMemcpyHtoDAsync(
            arena->address + arena_offsets_bytes[index], slot.address, sizes_bytes[index],
            reinterpret_cast<CUstream>(cuda_stream)
        );
        if (result == CUDA_SUCCESS) {
            result = cuEventRecord(slot.completion, reinterpret_cast<CUstream>(cuda_stream));
        }
        if (result != CUDA_SUCCESS) {
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            set_error("native shard ticket H2D", result);
            return DDO_VMM_CUDA_ERROR;
        }
        slot.in_flight = true;
    }
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_host_ring_read_file_to_arena_async(
    ddo_vmm_host_ring* ring,
    size_t slot_index,
    const char* path_utf8,
    uint64_t file_offset_bytes,
    ddo_vmm_model_arena* arena,
    size_t arena_offset_bytes,
    size_t size_bytes,
    void* cuda_stream
) {
#if !defined(_WIN32) && !defined(__linux__)
    (void)ring; (void)slot_index; (void)path_utf8; (void)file_offset_bytes;
    (void)arena; (void)arena_offset_bytes; (void)size_bytes; (void)cuda_stream;
    last_error = "native shard file transport is currently Windows-only";
    return DDO_VMM_INVALID_ARGUMENT;
#else
    if (ring == nullptr || arena == nullptr || path_utf8 == nullptr || slot_index >= ring->slots.size() ||
        size_bytes == 0 || size_bytes > ring->slot_size || arena_offset_bytes > arena->virtual_size ||
        size_bytes > arena->virtual_size - arena_offset_bytes || ring->device_ordinal != arena->device_ordinal) {
        last_error = "native shard transport arguments are invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    // Preserve this diagnostic convenience API, but route its source through
    // the same persistent, positioned native transport used by real tickets.
    // The wait is intentional here: callers requesting this combined API do
    // not expose a source-read ticket to overlap with other work.
    ddo_vmm_status status = ddo_vmm_host_ring_read_file_async(
        ring, slot_index, path_utf8, file_offset_bytes, size_bytes
    );
    if (status != DDO_VMM_SUCCESS) {
        return status;
    }
    status = ddo_vmm_host_ring_wait_file_read(ring, slot_index);
    if (status != DDO_VMM_SUCCESS) {
        return status;
    }
    return ddo_vmm_host_ring_copy_to_arena_async(
        ring, slot_index, 0, arena, arena_offset_bytes, size_bytes, cuda_stream
    );
#endif
}

extern "C" ddo_vmm_status ddo_vmm_host_ring_read_file_async(
    ddo_vmm_host_ring* ring, size_t slot_index, const char* path_utf8,
    uint64_t file_offset_bytes, size_t size_bytes
) {
#if !defined(_WIN32) && !defined(__linux__)
    (void)ring; (void)slot_index; (void)path_utf8; (void)file_offset_bytes; (void)size_bytes;
    last_error = "native shard file transport is currently Windows-only";
    return DDO_VMM_INVALID_ARGUMENT;
#else
    if (ring == nullptr || path_utf8 == nullptr || slot_index >= ring->slots.size() || size_bytes == 0 || size_bytes > ring->slot_size) {
        last_error = "native shard read arguments are invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(ring->mutex);
    auto& slot = ring->slots[slot_index];
    if (ring->closing || slot.read_busy() || (slot.in_flight && cuEventQuery(slot.completion) == CUDA_ERROR_NOT_READY)) {
        last_error = "native shard slot is not reusable";
        return DDO_VMM_OUT_OF_MEMORY;
    }
    const std::string path(path_utf8);
    auto file = ddo_source_io::invalid_file_handle;
    const auto existing = ring->shard_files.find(path);
    if (existing != ring->shard_files.end()) {
        file = existing->second;
    } else {
        file = ddo_source_io::open_file(path_utf8, last_error);
        if (file == ddo_source_io::invalid_file_handle) {
            return DDO_VMM_CUDA_ERROR;
        }
        ring->shard_files.emplace(path, file);
    }
    return enqueue_source_read_locked(ring, slot, file, file_offset_bytes, size_bytes);
#endif
}

#if defined(_WIN32) || defined(__linux__)
ddo_vmm_status ddo_source_io::read_source_async(
    ddo_vmm_host_ring* ring, size_t slot_index, const ddo_vmm_source_file* source,
    uint64_t offset, size_t size
) {
    if (!ring || !source || slot_index >= ring->slots.size() || !size || size > ring->slot_size)
        return DDO_VMM_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(ring->mutex);
    auto& slot = ring->slots[slot_index];
    if (ring->closing || slot.read_busy() || slot.in_flight) return DDO_VMM_BUSY;
    return enqueue_source_read_locked(ring, slot, source_handle(source), offset, size);
}
#endif

extern "C" ddo_vmm_status ddo_vmm_host_ring_wait_file_read(ddo_vmm_host_ring* ring, size_t slot_index) {
    if (ring == nullptr || slot_index >= ring->slots.size()) {
        last_error = "native shard read slot is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    std::unique_lock<std::mutex> lock(ring->mutex);
    auto& slot = ring->slots[slot_index];
    if (ring->closing) return DDO_VMM_BUSY;
    if (!slot.read_busy()) return DDO_VMM_SUCCESS;
    ddo_vmm_host_ring::FileReadResult result{};
    try {
#if defined(DDO_VMM_NATIVE_SOURCE_DISPATCHER)
        result = ddo_source_io::wait_read_without_ring_lock(
            slot, lock, ring->read_waiters_retired);
#else
        result = slot.file_read.get();
#endif
    } catch (const std::exception&) {
        slot.read_in_flight = false;
        last_error = "native shard asynchronous file read raised an exception";
        return DDO_VMM_CUDA_ERROR;
    }
    slot.read_in_flight = false;
    if (!result.succeeded) {
        last_error = result.error.empty() ? "native shard asynchronous file read failed" : result.error;
        return DDO_VMM_CUDA_ERROR;
    }
    return DDO_VMM_SUCCESS;
}

/* Source queue implementation is shared by the ring and persistent cache. */
#if defined(_WIN32) || defined(__linux__)
ddo_vmm_status enqueue_source_read_locked(
    ddo_vmm_host_ring* ring, ddo_vmm_host_ring::Slot& slot,
    ddo_source_io::file_handle file, uint64_t file_offset_bytes, size_t size_bytes
) {
    const size_t chunk_count = (size_bytes + kNativeSourceChunkBytes - 1) / kNativeSourceChunkBytes;
    auto completion = std::make_shared<ddo_vmm_host_ring::FileReadCompletion>();
    completion->pending_chunks.store(chunk_count, std::memory_order_release);
#if defined(DDO_VMM_NATIVE_SOURCE_DISPATCHER)
    slot.file_read = completion->promise.get_future().share();
#else
    slot.file_read = completion->promise.get_future();
#endif
    slot.read_in_flight = true;
    size_t source_offset = 0;
    try {
    while (source_offset < size_bytes) {
        const size_t chunk_size = std::min(kNativeSourceChunkBytes, size_bytes - source_offset);
        ddo_vmm_host_ring::FileReadTask task{};
        task.file = file;
        task.offset = file_offset_bytes + source_offset;
        task.destination = static_cast<unsigned char*>(slot.address) + source_offset;
        task.size = chunk_size;
        task.completion = completion;
        if (!enqueue_host_ring_source_task(ring, std::move(task))) {
            last_error = "native shard source queue is unavailable";
            while (source_offset < size_bytes) {
                const size_t abandoned_size = std::min(kNativeSourceChunkBytes, size_bytes - source_offset);
                complete_host_ring_source_read(completion, false, last_error);
                source_offset += abandoned_size;
            }
            return DDO_VMM_OUT_OF_MEMORY;
        }
        source_offset += chunk_size;
    }
    } catch (...) {
        // The successfully queued tasks still own their portions. Complete
        // only the unscheduled portions so wait/close cannot hang forever.
        last_error = "native shard source task allocation failed";
        while (source_offset < size_bytes) {
            const size_t chunk_size = std::min(kNativeSourceChunkBytes, size_bytes - source_offset);
            complete_host_ring_source_read(completion, false, last_error);
            source_offset += chunk_size;
        }
        return DDO_VMM_OUT_OF_MEMORY;
    }
    return DDO_VMM_SUCCESS;
}
#endif

extern "C" ddo_vmm_status ddo_vmm_host_ring_wait_slot_async(
    ddo_vmm_host_ring* ring,
    size_t slot_index,
    void* cuda_stream
) {
    if (ring == nullptr || slot_index >= ring->slots.size()) {
        last_error = "host ring wait slot is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(ring->mutex);
    const auto& slot = ring->slots[slot_index];
    if (!slot.in_flight) {
        return DDO_VMM_SUCCESS;
    }
    CUcontext temporary_primary_context = nullptr;
    if (!ensure_context(ring->device_ordinal, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    CUresult result = cuStreamWaitEvent(
        reinterpret_cast<CUstream>(cuda_stream), slot.completion, 0
    );
    release_temporary_primary_context(ring->device_ordinal, temporary_primary_context);
    if (result != CUDA_SUCCESS) {
        set_error("cuStreamWaitEvent(host ring)", result);
        return DDO_VMM_CUDA_ERROR;
    }
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_host_ring_copy_batch_to_arena_async(
    ddo_vmm_host_ring* ring,
    size_t slot_index,
    const size_t* slot_offsets_bytes,
    ddo_vmm_model_arena* arena,
    const size_t* arena_offsets_bytes,
    const size_t* sizes_bytes,
    size_t range_count,
    void* cuda_stream
) {
    if (ring == nullptr || arena == nullptr || slot_index >= ring->slots.size() ||
        slot_offsets_bytes == nullptr || arena_offsets_bytes == nullptr || sizes_bytes == nullptr ||
        range_count == 0 || ring->device_ordinal != arena->device_ordinal) {
        last_error = "host ring batch copy arguments are invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < range_count; ++index) {
        if (sizes_bytes[index] == 0 || slot_offsets_bytes[index] > ring->slot_size ||
            sizes_bytes[index] > ring->slot_size - slot_offsets_bytes[index] ||
            arena_offsets_bytes[index] > arena->virtual_size ||
            sizes_bytes[index] > arena->virtual_size - arena_offsets_bytes[index]) {
            last_error = "host ring batch copy range is outside its declared span";
            return DDO_VMM_INVALID_ARGUMENT;
        }
    }
    std::lock_guard<std::mutex> lock(ring->mutex);
    auto& slot = ring->slots[slot_index];
    if (slot.in_flight && cuEventQuery(slot.completion) == CUDA_ERROR_NOT_READY) {
        last_error = "host ring slot is still in flight";
        return DDO_VMM_OUT_OF_MEMORY;
    }
    slot.in_flight = false;
    CUcontext temporary_primary_context = nullptr;
    if (!activate_model_arena_context(arena, &temporary_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    for (size_t index = 0; index < range_count; ++index) {
        const auto* source = static_cast<const unsigned char*>(slot.address) + slot_offsets_bytes[index];
        CUresult result = cuMemcpyHtoDAsync(
            arena->address + arena_offsets_bytes[index], source, sizes_bytes[index],
            reinterpret_cast<CUstream>(cuda_stream)
        );
        if (result != CUDA_SUCCESS) {
            release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
            set_error("cuMemcpyHtoDAsync(host ring batch)", result);
            return DDO_VMM_CUDA_ERROR;
        }
    }
    CUresult result = cuEventRecord(slot.completion, reinterpret_cast<CUstream>(cuda_stream));
    release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
    if (result != CUDA_SUCCESS) {
        set_error("cuEventRecord(host ring batch)", result);
        return DDO_VMM_CUDA_ERROR;
    }
    slot.in_flight = true;
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_model_arena_unpin_range(
    ddo_vmm_model_arena* arena,
    size_t offset_bytes,
    size_t size_bytes
) {
    std::unique_lock<std::mutex> mutation_lock;
    if (arena != nullptr) {
        mutation_lock = std::unique_lock<std::mutex>(arena->mutation_mutex);
    }
    if (arena == nullptr || size_bytes == 0) {
        last_error = "model arena range must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    last_error.clear();
    if (offset_bytes > arena->virtual_size || size_bytes > arena->virtual_size - offset_bytes) {
        last_error = "model arena range is outside the reserved address span";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const size_t begin_page = model_arena_page_begin(arena, offset_bytes);
    const size_t end_page = model_arena_page_end(arena, offset_bytes, size_bytes);
    if (begin_page >= end_page || end_page > arena->pages.size()) {
        last_error = "model arena page range is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    for (size_t page_index = begin_page; page_index < end_page; ++page_index) {
        auto& page = arena->pages[page_index];
        if (page.pin_count > 0) {
            page.pin_count -= 1;
        }
    }
    return DDO_VMM_SUCCESS;
}

extern "C" uintptr_t ddo_vmm_model_arena_address(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : static_cast<uintptr_t>(arena->address);
}

extern "C" int ddo_vmm_model_arena_device_ordinal(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? -1 : arena->device_ordinal;
}

extern "C" size_t ddo_vmm_model_arena_virtual_size(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->virtual_size;
}

extern "C" size_t ddo_vmm_model_arena_page_size(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->page_size;
}

extern "C" size_t ddo_vmm_model_arena_resident_size(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->resident_pages * arena->page_size;
}

extern "C" size_t ddo_vmm_model_arena_fault_ceiling(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->fault_ceiling_page * arena->page_size;
}

extern "C" uint64_t ddo_vmm_model_arena_materialized_pages(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->materialized_pages;
}

extern "C" uint64_t ddo_vmm_model_arena_reused_pages(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->reused_pages;
}

extern "C" uint64_t ddo_vmm_model_arena_evicted_pages(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->evicted_pages;
}

extern "C" size_t ddo_vmm_model_arena_resident_prefix_pages(const ddo_vmm_model_arena* arena) {
    if (arena == nullptr) {
        return 0;
    }
    size_t prefix_pages = 0;
    while (prefix_pages < arena->pages.size() && arena->pages[prefix_pages].allocation != 0) {
        ++prefix_pages;
    }
    return prefix_pages;
}

extern "C" uint64_t ddo_vmm_model_arena_tail_evictions(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->tail_evictions;
}

extern "C" uint64_t ddo_vmm_model_arena_pressure_evictions(const ddo_vmm_model_arena* arena) {
    return arena == nullptr ? 0 : arena->pressure_evictions;
}

extern "C" ddo_vmm_status ddo_vmm_host_ring_create(
    int device_ordinal,
    size_t slot_size_bytes,
    size_t slot_count,
    ddo_vmm_host_ring** out_ring
) {
    if (out_ring == nullptr || slot_size_bytes == 0 || slot_count == 0) {
        last_error = "host ring output, slot size, and slot count must be valid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    *out_ring = nullptr;
    // Legacy staging remains admissible independently of experimental cache
    // caps. Serialize registration so cache cannot admit against a stale total.
    std::lock_guard<std::mutex> accounting_lock(ddo_host_accounting::mutex);
    if (slot_count > (std::numeric_limits<uint64_t>::max()
            - ddo_host_accounting::staging_registered_bytes) / slot_size_bytes) {
        last_error = "host ring accounting size overflow";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    const uint64_t requested_bytes = slot_size_bytes * slot_count;
    const uint64_t cache_limit = ddo_host_accounting::active_cache_limit_bytes;
    if (cache_limit != std::numeric_limits<uint64_t>::max()) {
        const uint64_t cache_bytes = ddo_host_accounting::cache_committed_bytes;
        const uint64_t staging_bytes = ddo_host_accounting::staging_registered_bytes;
        if (cache_bytes > cache_limit || staging_bytes > cache_limit - cache_bytes
            || requested_bytes > cache_limit - cache_bytes - staging_bytes) {
            last_error = "host ring would exceed the active source-cache process limit";
            return DDO_VMM_OUT_OF_MEMORY;
        }
        ddo_host_platform::Headroom memory{};
        const uint64_t headroom = ddo_host_accounting::active_cache_headroom_bytes;
        if (!ddo_host_platform::headroom(memory)
            || !ddo_host_platform::permits(memory, requested_bytes, headroom)) {
            last_error = "host ring registration would breach source-cache RAM/commit headroom";
            return DDO_VMM_OUT_OF_MEMORY;
        }
    }
#if defined(__linux__)
    else {
        // Ring-only transport must not bypass Linux process/container limits.
        ddo_host_platform::Headroom memory{};
        if (!ddo_host_platform::headroom(memory)
            || !ddo_host_platform::permits(memory, requested_bytes, 0)) {
            last_error = "host ring Linux host/cgroup headroom unavailable or insufficient";
            return DDO_VMM_OUT_OF_MEMORY;
        }
    }
#endif
    CUcontext retained_primary_context = nullptr;
    if (!ensure_context(device_ordinal, &retained_primary_context)) {
        return DDO_VMM_CUDA_ERROR;
    }
    auto ring = std::make_unique<ddo_vmm_host_ring>();
    ring->device_ordinal = device_ordinal;
    ring->slot_size = slot_size_bytes;
    ring->slots.reserve(slot_count);
    for (size_t index = 0; index < slot_count; ++index) {
        ddo_vmm_host_ring::Slot slot{};
#if defined(_WIN32) || defined(__linux__)
        slot.address = ddo_host_platform::allocate(slot_size_bytes);
        if (slot.address != nullptr
            && checked(cuMemHostRegister(slot.address, slot_size_bytes, CU_MEMHOSTREGISTER_PORTABLE), "cuMemHostRegister")) {
            slot.virtual_memory_registered = true;
        }
        if (!slot.virtual_memory_registered ||
#else
        if (!checked(cuMemHostAlloc(&slot.address, slot_size_bytes, CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc") ||
#endif
            !checked(cuEventCreate(&slot.completion, CU_EVENT_DISABLE_TIMING), "cuEventCreate(host ring)")) {
            if (slot.completion != nullptr) {
                cuEventDestroy(slot.completion);
            }
            if (slot.address != nullptr) {
#if defined(_WIN32) || defined(__linux__)
                if (slot.virtual_memory_registered) {
                    cuMemHostUnregister(slot.address);
                }
                ddo_host_platform::release(slot.address, slot_size_bytes);
#else
                cuMemFreeHost(slot.address);
#endif
            }
            for (const auto& allocated : ring->slots) {
                cuEventDestroy(allocated.completion);
#if defined(_WIN32) || defined(__linux__)
                cuMemHostUnregister(allocated.address);
                ddo_host_platform::release(allocated.address, slot_size_bytes);
#else
                cuMemFreeHost(allocated.address);
#endif
            }
            release_temporary_primary_context(device_ordinal, retained_primary_context);
            return DDO_VMM_CUDA_ERROR;
        }
        ring->slots.push_back(std::move(slot));
    }
#if defined(_WIN32) || defined(__linux__)
    if (!start_host_ring_source_workers(ring.get())) {
        for (const auto& allocated : ring->slots) {
            cuEventDestroy(allocated.completion);
            cuMemHostUnregister(allocated.address);
            ddo_host_platform::release(allocated.address, slot_size_bytes);
        }
        release_temporary_primary_context(device_ordinal, retained_primary_context);
        return DDO_VMM_CUDA_ERROR;
    }
#endif
    release_temporary_primary_context(device_ordinal, retained_primary_context);
    ring->accounted_staging = true;
    ddo_host_accounting::staging_registered_bytes += slot_size_bytes * slot_count;
    *out_ring = ring.release();
    return DDO_VMM_SUCCESS;
}

extern "C" void ddo_vmm_host_ring_destroy(ddo_vmm_host_ring* ring) {
    if (ring == nullptr) {
        return;
    }
    std::unique_lock<std::mutex> lock(ring->mutex);
    ring->closing = true;
    ring->read_waiters_retired.wait(lock, [&] {
        return std::all_of(ring->slots.begin(), ring->slots.end(),
                           [](const auto& slot) { return slot.read_waiters == 0; });
    });
    uint64_t released_staging = 0;
    for (const auto& slot : ring->slots) {
        if (slot.read_in_flight) {
            slot.file_read.wait();
        }
        if (slot.in_flight) {
            cuEventSynchronize(slot.completion);
        }
        if (slot.completion) cuEventDestroy(slot.completion);
#if defined(_WIN32) || defined(__linux__)
        if (slot.virtual_memory_registered) {
            if (cuMemHostUnregister(slot.address) == CUDA_SUCCESS && ring->accounted_staging)
                released_staging += ring->slot_size;
        }
        if (slot.address) ddo_host_platform::release(slot.address, ring->slot_size);
#else
        if (slot.address && cuMemFreeHost(slot.address) == CUDA_SUCCESS && ring->accounted_staging)
            released_staging += ring->slot_size;
#endif
    }
#if defined(_WIN32) || defined(__linux__)
    stop_host_ring_source_workers(ring);
    for (const auto& [path, file] : ring->shard_files) {
        (void)path;
        ddo_source_io::close_file(file);
    }
    ring->shard_files.clear();
#endif
    lock.unlock();  // The mutex must outlive its lock guard.
    {
        std::lock_guard<std::mutex> accounting_lock(ddo_host_accounting::mutex);
        ddo_host_accounting::staging_registered_bytes -= released_staging;
    }
    delete ring;
}

#if defined(_WIN32) || defined(__linux__)
ddo_vmm_status ddo_source_io::create_lazy_ring(
    int device, size_t slot_bytes, size_t slots, ddo_vmm_host_ring** output
) {
    if (!output || !slot_bytes || !slots) return DDO_VMM_INVALID_ARGUMENT;
    *output = nullptr;
    auto ring = std::make_unique<ddo_vmm_host_ring>();
    ring->device_ordinal = device;
    ring->slot_size = slot_bytes;
    ring->slots.resize(slots);
    if (!start_host_ring_source_workers(ring.get())) return DDO_VMM_INTERNAL_ERROR;
    *output = ring.release();
    return DDO_VMM_SUCCESS;
}

bool ddo_source_io::source_read_complete(ddo_vmm_host_ring* ring, size_t index) {
    if (!ring || index >= ring->slots.size()) return true;
    std::lock_guard<std::mutex> lock(ring->mutex);
    const auto& slot = ring->slots[index];
    if (!slot.read_in_flight) return true;
    // An exceptional/invalid completion must reach the draining error path,
    // never masquerade as successful entry readiness.
    return !slot.file_read.valid()
        || slot.file_read.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

ddo_vmm_status ddo_source_io::commit_storage(ddo_vmm_host_ring* ring, size_t index) {
    if (!ring || index >= ring->slots.size()) return DDO_VMM_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(ring->mutex);
    auto& slot = ring->slots[index];
    if (ring->closing) return DDO_VMM_BUSY;
    if (slot.address) return DDO_VMM_SUCCESS;
    CUevent event = nullptr;
    if (!checked(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "cuEventCreate(cache slot)"))
        return DDO_VMM_CUDA_ERROR;
    void* address = ddo_host_platform::allocate(ring->slot_size);
    if (!address) {
        cuEventDestroy(event);
        last_error = "progressive host slot commit failed";
        return DDO_VMM_OUT_OF_MEMORY;
    }
    slot.address = address;
    slot.completion = event;
    return DDO_VMM_SUCCESS;
}

ddo_vmm_status ddo_source_io::register_storage(ddo_vmm_host_ring* ring, size_t index) {
    if (!ring || index >= ring->slots.size()) return DDO_VMM_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(ring->mutex);
    auto& slot = ring->slots[index];
    if (ring->closing || !slot.address || slot.read_busy() || slot.in_flight) return DDO_VMM_BUSY;
    if (slot.virtual_memory_registered) return DDO_VMM_SUCCESS;
    const auto result = cuMemHostRegister(slot.address, ring->slot_size, CU_MEMHOSTREGISTER_PORTABLE);
    if (!checked(result, "cuMemHostRegister(persistent storage)"))
        return result == CUDA_ERROR_OUT_OF_MEMORY ? DDO_VMM_OUT_OF_MEMORY : DDO_VMM_CUDA_ERROR;
    slot.virtual_memory_registered = true;
    return DDO_VMM_SUCCESS;
}

ddo_vmm_status ddo_source_io::unregister_storage(ddo_vmm_host_ring* ring, size_t index) {
    if (!ring || index >= ring->slots.size()) return DDO_VMM_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(ring->mutex);
    auto& slot = ring->slots[index];
    if (ring->closing || slot.read_busy() || slot.in_flight) return DDO_VMM_BUSY;
    if (slot.virtual_memory_registered) {
        if (!checked(cuMemHostUnregister(slot.address), "cuMemHostUnregister(persistent storage)"))
            return DDO_VMM_CUDA_ERROR;
        slot.virtual_memory_registered = false;
    }
    return DDO_VMM_SUCCESS;
}

ddo_vmm_status ddo_source_io::commit_slot(ddo_vmm_host_ring* ring, size_t index) {
    auto status = commit_storage(ring, index);
    if (status != DDO_VMM_SUCCESS) return status;
    status = register_storage(ring, index);
    if (status != DDO_VMM_SUCCESS) release_slot(ring, index);
    return status;
}

ddo_vmm_status ddo_source_io::stage_host_bytes(ddo_vmm_host_ring* ring, size_t index, const void* bytes, size_t size) {
    if (!ring || index >= ring->slots.size() || !bytes || !size || size > ring->slot_size)
        return DDO_VMM_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(ring->mutex);
    auto& slot = ring->slots[index];
    if (ring->closing || !slot.address || !slot.virtual_memory_registered || slot.read_busy())
        return DDO_VMM_BUSY;
    if (slot.in_flight) {
        const auto result = cuEventQuery(slot.completion);
        if (result == CUDA_ERROR_NOT_READY) return DDO_VMM_BUSY;
        if (!checked(result, "cuEventQuery(retained-source fallback slot)")) return DDO_VMM_CUDA_ERROR;
        slot.in_flight = false;
    }
    std::memcpy(slot.address, bytes, size);
    return DDO_VMM_SUCCESS;
}

ddo_vmm_status ddo_source_io::release_slot(ddo_vmm_host_ring* ring, size_t index) {
    if (!ring || index >= ring->slots.size()) return DDO_VMM_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(ring->mutex);
    auto& slot = ring->slots[index];
    if (ring->closing || slot.read_busy() || slot.in_flight) return DDO_VMM_BUSY;
    if (slot.completion) {
        if (!checked(cuEventDestroy(slot.completion), "cuEventDestroy(cache slot)")) return DDO_VMM_CUDA_ERROR;
        slot.completion = nullptr;
    }
    if (slot.virtual_memory_registered) {
        if (!checked(cuMemHostUnregister(slot.address), "cuMemHostUnregister(cache slot)")) return DDO_VMM_CUDA_ERROR;
        slot.virtual_memory_registered = false;
    }
    if (slot.address) {
        if (!ddo_host_platform::release(slot.address, ring->slot_size)) {
            last_error = "progressive host slot release failed";
            return DDO_VMM_INTERNAL_ERROR;
        }
        slot.address = nullptr;
    }
    return DDO_VMM_SUCCESS;
}
#endif

extern "C" size_t ddo_vmm_host_ring_slot_size(const ddo_vmm_host_ring* ring) {
    return ring == nullptr ? 0 : ring->slot_size;
}

extern "C" size_t ddo_vmm_host_ring_slot_count(const ddo_vmm_host_ring* ring) {
    return ring == nullptr ? 0 : ring->slots.size();
}

extern "C" size_t ddo_vmm_host_ring_source_worker_count(const ddo_vmm_host_ring* ring) {
#if defined(_WIN32) || defined(__linux__)
    return ring == nullptr ? 0 : ring->source_workers.size();
#else
    (void)ring;
    return 0;
#endif
}

extern "C" uint64_t ddo_vmm_host_ring_source_read_chunks(const ddo_vmm_host_ring* ring) {
#if defined(_WIN32) || defined(__linux__)
    return ring == nullptr ? 0 : ring->source_read_chunks.load(std::memory_order_relaxed);
#else
    (void)ring;
    return 0;
#endif
}

extern "C" uint64_t ddo_vmm_host_ring_source_read_bytes(const ddo_vmm_host_ring* ring) {
#if defined(_WIN32) || defined(__linux__)
    return ring == nullptr ? 0 : ring->source_read_bytes.load(std::memory_order_relaxed);
#else
    (void)ring;
    return 0;
#endif
}

extern "C" uint64_t ddo_vmm_host_ring_source_read_failures(const ddo_vmm_host_ring* ring) {
#if defined(_WIN32) || defined(__linux__)
    return ring == nullptr ? 0 : ring->source_read_failures.load(std::memory_order_relaxed);
#else
    (void)ring;
    return 0;
#endif
}

extern "C" uintptr_t ddo_vmm_host_ring_slot_address(const ddo_vmm_host_ring* ring, size_t slot_index) {
    if (ring == nullptr || slot_index >= ring->slots.size()) {
        return 0;
    }
    return reinterpret_cast<uintptr_t>(ring->slots[slot_index].address);
}

extern "C" int ddo_vmm_host_ring_slot_ready(ddo_vmm_host_ring* ring, size_t slot_index) {
    if (ring == nullptr || slot_index >= ring->slots.size()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(ring->mutex);
    auto& slot = ring->slots[slot_index];
    if (slot.read_busy()) {
        return 0;
    }
    if (!slot.in_flight) {
        return 1;
    }
    CUresult result = cuEventQuery(slot.completion);
    if (result == CUDA_SUCCESS) {
        slot.in_flight = false;
        return 1;
    }
    if (result == CUDA_ERROR_NOT_READY) {
        return 0;
    }
    set_error("cuEventQuery(host ring)", result);
    return 0;
}

extern "C" ddo_vmm_status ddo_vmm_host_ring_wait_slot(ddo_vmm_host_ring* ring, size_t slot_index) {
    if (ring == nullptr || slot_index >= ring->slots.size()) {
        last_error = "host ring wait slot is invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(ring->mutex);
    auto& slot = ring->slots[slot_index];
    if (slot.read_busy()) {
        last_error = "host ring source read must finish before slot reuse";
        return DDO_VMM_OUT_OF_MEMORY;
    }
    if (!slot.in_flight) {
        return DDO_VMM_SUCCESS;
    }
    CUresult result = cuEventSynchronize(slot.completion);
    if (result != CUDA_SUCCESS) {
        set_error("cuEventSynchronize(host ring)", result);
        return DDO_VMM_CUDA_ERROR;
    }
    slot.in_flight = false;
    return DDO_VMM_SUCCESS;
}

extern "C" void ddo_vmm_pressure_register(ddo_vmm_region* region) {
    if (region == nullptr) {
        return;
    }
    std::scoped_lock lock(pressure_mutex);
    if (pressure_entry(region) == nullptr) {
        pressure_entries.push_back({region, false, ++pressure_clock});
    }
}

extern "C" void ddo_vmm_pressure_unregister(ddo_vmm_region* region) {
    std::scoped_lock lock(pressure_mutex);
    for (auto entry = pressure_entries.begin(); entry != pressure_entries.end(); ++entry) {
        if (entry->region == region) {
            pressure_entries.erase(entry);
            return;
        }
    }
}

extern "C" void ddo_vmm_pressure_set_pinned(ddo_vmm_region* region, int pinned) {
    std::scoped_lock lock(pressure_mutex);
    if (PressureEntry* entry = pressure_entry(region)) {
        entry->pinned = pinned != 0;
        entry->touch = ++pressure_clock;
    }
}

extern "C" void ddo_vmm_pressure_touch(ddo_vmm_region* region) {
    std::scoped_lock lock(pressure_mutex);
    if (PressureEntry* entry = pressure_entry(region)) {
        entry->touch = ++pressure_clock;
    }
}

extern "C" int ddo_vmm_pressure_evict_one(int device_ordinal) {
    std::scoped_lock lock(pressure_mutex);
    // This is reached from the allocator that owns ordinary PyTorch CUDA
    // allocations.  Release an eligible model page first, retaining the
    // arena's virtual address and its contiguous low-address prefix. A
    // try-lock avoids waiting behind the exact model operation that may have
    // initiated the allocation request.
    for (const ModelArenaPressureEntry& entry : pressure_model_arenas) {
        ddo_vmm_model_arena* arena = entry.arena;
        if (arena == nullptr || arena->device_ordinal != device_ordinal) {
            continue;
        }
        std::unique_lock<std::mutex> arena_lock(arena->mutation_mutex, std::try_to_lock);
        if (!arena_lock.owns_lock() || arena->resident_pages == 0) {
            continue;
        }
        CUcontext temporary_primary_context = nullptr;
        if (!activate_model_arena_context(arena, &temporary_primary_context)) {
            continue;
        }
        const bool evicted = model_arena_evict_one(arena);
        release_temporary_primary_context(arena->device_ordinal, temporary_primary_context);
        if (evicted) {
            arena->pressure_evictions += 1;
            return 1;
        }
    }
    PressureEntry* selected = nullptr;
    for (PressureEntry& entry : pressure_entries) {
        if (
            entry.region == nullptr || entry.pinned || entry.region->device_ordinal != device_ordinal
            || region_resident_size(entry.region) == 0
        ) {
            continue;
        }
        if (selected == nullptr || entry.touch < selected->touch) {
            selected = &entry;
        }
    }
    if (selected == nullptr) {
        return 0;
    }
    if (ddo_vmm_region_evict(selected->region) != DDO_VMM_SUCCESS) {
        return 0;
    }
    selected->touch = ++pressure_clock;
    return 1;
}

extern "C" ddo_vmm_status ddo_vmm_pressure_prepare_cuda_allocation(
    int device_ordinal,
    size_t required_bytes
) {
    if (device_ordinal < 0 || required_bytes == 0) {
        last_error = "CUDA allocation pressure arguments are invalid";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    // With no configured WDDM reserve this is deliberately a no-op. The
    // allocation path still has its OOM retry below, avoiding speculative
    // eviction during ordinary steady-state work.
    if (!make_dxgi_budget_room(
        device_ordinal, required_bytes, DDO_VMM_BACKING_DEVICE, false
    )) {
        last_error = "native VMM could not restore CUDA allocation headroom";
        return DDO_VMM_OUT_OF_MEMORY;
    }
    return DDO_VMM_SUCCESS;
}

extern "C" ddo_vmm_status ddo_vmm_windows_memory_query(
    int device_ordinal, ddo_vmm_windows_memory_info* out_info
) {
    if (device_ordinal < 0 || out_info == nullptr) {
        return DDO_VMM_INVALID_ARGUMENT;
    }
#if defined(_WIN32)
    if (!checked(cuInit(0), "cuInit")) {
        return DDO_VMM_CUDA_ERROR;
    }
    if (query_windows_memory(device_ordinal, out_info)) {
        return DDO_VMM_SUCCESS;
    }
    last_error = "unable to query DXGI memory budget for CUDA device";
    return DDO_VMM_CUDA_ERROR;
#else
    (void)out_info;
    last_error = "Windows DXGI memory budgets are unavailable on this platform";
    return DDO_VMM_INVALID_ARGUMENT;
#endif
}

extern "C" ddo_vmm_status ddo_vmm_dxgi_budget_guard_set(
    int device_ordinal,
    uint64_t local_reserve_bytes,
    uint64_t shared_reserve_bytes,
    uint64_t system_reserve_bytes
) {
    if (device_ordinal < 0) {
        return DDO_VMM_INVALID_ARGUMENT;
    }
    std::scoped_lock lock(dxgi_budget_guard_mutex);
    for (auto& [ordinal, config] : dxgi_budget_guards) {
        if (ordinal == device_ordinal) {
            config = {local_reserve_bytes, shared_reserve_bytes, system_reserve_bytes};
            return DDO_VMM_SUCCESS;
        }
    }
    dxgi_budget_guards.push_back({
        device_ordinal,
        {local_reserve_bytes, shared_reserve_bytes, system_reserve_bytes},
    });
    return DDO_VMM_SUCCESS;
}

extern "C" int ddo_vmm_model_arena_enable_streaming(
    ddo_vmm_model_arena* arena, size_t execution_buffer_bytes
) {
    if (!arena || !execution_buffer_bytes) return DDO_VMM_INVALID_ARGUMENT;
    std::scoped_lock lock(arena->mutation_mutex);
    const size_t pages = execution_buffer_bytes / arena->page_size
        + (execution_buffer_bytes % arena->page_size != 0);
    const size_t budget_pages = arena->resident_budget / arena->page_size;
    if (arena->streaming_admission || arena->resident_pages || pages >= budget_pages) {
        last_error = "execution buffer requires unused arena and room within residency ceiling";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    arena->resident_budget = (budget_pages - pages) * arena->page_size;
    arena->fault_ceiling_page = std::min(arena->pages.size(), budget_pages - pages);
    arena->admission_limit_reason = "native streaming limit";
    arena->streaming_admission = true;
    return DDO_VMM_SUCCESS;
}

extern "C" int ddo_vmm_model_arena_record_compute(ddo_vmm_model_arena* arena, void* stream) {
    if (!arena) return DDO_VMM_INVALID_ARGUMENT;
    std::scoped_lock lock(arena->mutation_mutex);
    CUcontext temporary = nullptr;
    if (!activate_model_arena_context(arena, &temporary)) return DDO_VMM_CUDA_ERROR;
    bool ok = true;
    if (!arena->compute_fence)
        ok = checked(cuEventCreate(&arena->compute_fence, CU_EVENT_DISABLE_TIMING), "execution event create");
    if (ok) ok = checked(cuEventRecord(arena->compute_fence, static_cast<CUstream>(stream)), "execution event record");
    if (ok) arena->compute_fence_recorded = true;
    release_temporary_primary_context(arena->device_ordinal, temporary);
    return ok ? DDO_VMM_SUCCESS : DDO_VMM_CUDA_ERROR;
}

extern "C" int ddo_vmm_model_arena_wait_compute(ddo_vmm_model_arena* arena, void* stream) {
    if (!arena) return DDO_VMM_INVALID_ARGUMENT;
    std::scoped_lock lock(arena->mutation_mutex);
    if (!arena->compute_fence_recorded) return DDO_VMM_SUCCESS;
    CUcontext temporary = nullptr;
    if (!activate_model_arena_context(arena, &temporary)) return DDO_VMM_CUDA_ERROR;
    bool ok;
    {
        ddo_vmm_detail::TimelineScope timing(3, reinterpret_cast<uintptr_t>(arena), 0, stream);
        ok = checked(cuStreamWaitEvent(static_cast<CUstream>(stream), arena->compute_fence, 0), "execution buffer reuse wait");
    }
    release_temporary_primary_context(arena->device_ordinal, temporary);
    return ok ? DDO_VMM_SUCCESS : DDO_VMM_CUDA_ERROR;
}

extern "C" int ddo_vmm_model_arena_drain_compute(ddo_vmm_model_arena* arena) {
    if (!arena) return DDO_VMM_INVALID_ARGUMENT;
    std::scoped_lock lock(arena->mutation_mutex);
    if (!arena->compute_fence_recorded) return DDO_VMM_SUCCESS;
    CUcontext temporary = nullptr;
    if (!activate_model_arena_context(arena, &temporary)) return DDO_VMM_CUDA_ERROR;
    const bool ok = checked(cuEventSynchronize(arena->compute_fence), "execution buffer close wait");
    release_temporary_primary_context(arena->device_ordinal, temporary);
    return ok ? DDO_VMM_SUCCESS : DDO_VMM_CUDA_ERROR;
}

extern "C" int ddo_vmm_execution_pool_create(ddo_vmm_model_arena* model,
    size_t slot_bytes, size_t slot_count, ddo_vmm_execution_pool** output) {
    if (!output) return DDO_VMM_INVALID_ARGUMENT;
    *output = nullptr;
    if (!model || !slot_bytes || slot_count < 2 || slot_count > 4)
        return DDO_VMM_INVALID_ARGUMENT;
    const size_t capacity = align_up(slot_bytes, model->page_size);
    if (!capacity || capacity > model->resident_budget / slot_count
        || capacity * slot_count >= model->resident_budget) {
        last_error = "execution slots must fit inside existing weight residency ceiling";
        return DDO_VMM_INVALID_ARGUMENT;
    }
    try {
        auto pool = std::make_unique<ddo_vmm_execution_pool>();
        pool->slots.resize(slot_count);
        for (auto& slot : pool->slots) {
            auto status = ddo_vmm_model_arena_create(model->device_ordinal, capacity,
                capacity, model->page_size, model->backing_kind, &slot.arena);
            if (status != DDO_VMM_SUCCESS) {
                return status;
            }
        }
        auto status = ddo_vmm_model_arena_enable_streaming(model, capacity * slot_count);
        if (status != DDO_VMM_SUCCESS) {
            return status;
        }
        *output = pool.release();
        return DDO_VMM_SUCCESS;
    } catch (...) {
        last_error = "execution slot metadata allocation failed";
        return DDO_VMM_OUT_OF_MEMORY;
    }
}

extern "C" int ddo_vmm_execution_pool_acquire(ddo_vmm_execution_pool* pool,
    void* upload_stream, size_t* index, ddo_vmm_model_arena** arena) {
    if (!pool || !index || !arena) return DDO_VMM_INVALID_ARGUMENT;
    *arena = nullptr;
    std::scoped_lock lock(pool->mutex);
    for (size_t step = 0; step < pool->slots.size(); ++step) {
        const size_t candidate = (pool->next + step) % pool->slots.size();
        auto& slot = pool->slots[candidate];
        if (slot.leased) continue;
        auto status = ddo_vmm_model_arena_wait_compute(slot.arena, upload_stream);
        if (status != DDO_VMM_SUCCESS) return status;
        if (!slot.materialized) {
            uint64_t signature = 0;
            status = ddo_vmm_model_arena_fault_range(slot.arena, 0,
                slot.arena->virtual_size, &signature);
            if (status != DDO_VMM_SUCCESS) return status;
            slot.materialized = true; // Permanently pin; only close releases.
        }
        slot.leased = true;
        pool->next = (candidate + 1) % pool->slots.size();
        *index = candidate;
        *arena = slot.arena;
        return DDO_VMM_SUCCESS;
    }
    last_error = "all native execution slots still have live consumers";
    return DDO_VMM_BUSY;
}

extern "C" int ddo_vmm_execution_pool_release(ddo_vmm_execution_pool* pool,
    size_t index, void* last_use_stream) {
    if (!pool || index >= pool->slots.size()) return DDO_VMM_INVALID_ARGUMENT;
    std::scoped_lock lock(pool->mutex);
    auto& slot = pool->slots[index];
    if (!slot.leased) return DDO_VMM_INVALID_ARGUMENT;
    auto status = ddo_vmm_model_arena_record_compute(slot.arena, last_use_stream);
    if (status == DDO_VMM_SUCCESS) slot.leased = false;
    return status; // On event failure the lease remains owned, never reusable.
}

extern "C" int ddo_vmm_execution_pool_close(ddo_vmm_execution_pool** owner) {
    if (!owner) return DDO_VMM_INVALID_ARGUMENT;
    auto* pool = *owner;
    if (!pool) return DDO_VMM_SUCCESS;
    {
        std::scoped_lock lock(pool->mutex);
        for (const auto& slot : pool->slots) {
            if (slot.leased) {
                last_error = "execution pool still has live slot leases";
                return DDO_VMM_BUSY;
            }
            auto status = ddo_vmm_model_arena_drain_compute(slot.arena);
            if (status != DDO_VMM_SUCCESS) return status;
        }
        for (auto& slot : pool->slots) {
            ddo_vmm_model_arena_destroy(slot.arena);
            slot.arena = nullptr;
        }
    }
    delete pool;
    *owner = nullptr;
    return DDO_VMM_SUCCESS;
}

extern "C" const char* ddo_vmm_last_error(void) {
    return last_error.c_str();
}
