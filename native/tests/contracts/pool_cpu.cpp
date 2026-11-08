// Compile the actual isolated pool implementation with in-process CUDA stubs.
#include "native/src/ddo_vmm_torch_pool.cpp"
#include "dead_pages.h"
#include "factory.h"
#include <array>
#include <functional>
#include <iostream>
#include <map>
#include <windows.h>

constexpr size_t fixture_page = 1024;
constexpr uintptr_t fixture_base = 0x100000;
int checks = 0;
void check(bool value, const char* text) { ++checks; if (!value) throw std::runtime_error(text); }
CUcontext fixture_context() { return reinterpret_cast<CUcontext>(uintptr_t{0x100}); }
CUstream fixture_stream() { return reinterpret_cast<CUstream>(uintptr_t{0x200}); }
bool events_complete = false, dependency_failure = false, query_failure = false;
size_t region_attempts = 0;
ddo_vmm_status region_status = DDO_VMM_OUT_OF_MEMORY;
ddo_vmm_status pressure_status = DDO_VMM_SUCCESS;
size_t pressure_calls = 0, last_pressure_bytes = 0;
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
std::function<ddo_vmm_status(size_t)> auto_pressure;
size_t auto_sync_calls = 0;
CUresult fixture_barrier_status = CUDA_SUCCESS;
std::function<void()> fixture_barrier_callback;
CUcontext fixture_current_context = fixture_context();
#endif
std::map<CUevent, bool> fixture_events;
CUresult CUDAAPI cuCtxGetCurrent(CUcontext* output) { *output = fixture_current_context; return CUDA_SUCCESS; }
CUresult CUDAAPI cuCtxGetDevice(CUdevice* output) { *output = 0; return CUDA_SUCCESS; }
CUresult CUDAAPI cuCtxPushCurrent(CUcontext) { return CUDA_SUCCESS; }
CUresult CUDAAPI cuCtxPopCurrent(CUcontext* output) { *output = fixture_context(); return CUDA_SUCCESS; }
CUresult CUDAAPI cuStreamGetCtx(CUstream, CUcontext* output) { *output = fixture_context(); return CUDA_SUCCESS; }
CUresult CUDAAPI cuStreamGetId(CUstream input, unsigned long long* output) {
    *output = reinterpret_cast<uintptr_t>(input); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventCreate(CUevent* output, unsigned int) {
    *output = reinterpret_cast<CUevent>(uintptr_t{0x1000} + fixture_events.size());
    fixture_events[*output] = true; return CUDA_SUCCESS;
}
CUresult CUDAAPI cuEventDestroy(CUevent event) { fixture_events.erase(event); return CUDA_SUCCESS; }
CUresult CUDAAPI cuEventRecord(CUevent, CUstream) { return CUDA_SUCCESS; }
CUresult CUDAAPI cuEventQuery(CUevent) {
    return query_failure ? CUDA_ERROR_ILLEGAL_ADDRESS : (events_complete ? CUDA_SUCCESS : CUDA_ERROR_NOT_READY);
}
// Fence-wait double. Default keeps the inherited "never completes" behavior;
// a successful wait completes the fixture's events, like a real fence wait.
CUresult fixture_event_sync_status = CUDA_ERROR_NOT_READY;
size_t event_sync_calls = 0, ctx_sync_calls = 0;
bool sync_lock_violation = false;
CUresult CUDAAPI cuEventSynchronize(CUevent) {
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
    ++auto_sync_calls;
#endif
    ++event_sync_calls;
    if (!pool_mutex.try_lock()) sync_lock_violation = true; else pool_mutex.unlock();
    if (fixture_event_sync_status == CUDA_SUCCESS) events_complete = true;
    return fixture_event_sync_status;
}
CUstreamCaptureStatus fixture_capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
CUresult fixture_capture_query = CUDA_SUCCESS;
CUresult CUDAAPI cuStreamIsCapturing(CUstream, CUstreamCaptureStatus* output) {
    *output = fixture_capture_status; return fixture_capture_query;
}
CUresult CUDAAPI cuCtxSynchronize() {
    ++auto_sync_calls; ++ctx_sync_calls;
    check(pool_mutex.try_lock(), "Context barrier outside pool lock"); pool_mutex.unlock();
    if (fixture_barrier_callback) { auto callback=std::move(fixture_barrier_callback); callback(); }
    return fixture_barrier_status;
}
CUresult CUDAAPI cuStreamWaitEvent(CUstream, CUevent, unsigned int) {
    return dependency_failure ? CUDA_ERROR_INVALID_CONTEXT : CUDA_SUCCESS;
}
namespace ddo_vmm_detail { int recheck_execution_backing(int) { return 0; } }
#if defined(DDO_VMM_TIMING_COUNTERS) && defined(_WIN32)
// Windows memory observation double: configurable values (defaults below are
// what the timing suite expects), or a failed query.
bool fixture_memory_query_ok = true;
size_t fixture_memory_queries = 0;
uint64_t fixture_local_budget = 8000, fixture_local_usage = 7000;
bool query_windows_memory(int, ddo_vmm_windows_memory_info* out) {
    ++fixture_memory_queries;
    if (!pool_mutex.try_lock()) throw std::runtime_error("memory query under pool lock");
    pool_mutex.unlock();
    if (!fixture_memory_query_ok) return false;
    *out = {fixture_local_budget, fixture_local_usage, 32000, 1000, 40000, 50000};
    return true;
}
#endif
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
// Model-arena doubles: a fixed number of evictable pages of fixture size.
size_t arena_pages = 0, arena_busy = 0, arena_count_calls = 0, arena_evict_calls = 0;
size_t arena_page_bytes = fixture_page;
size_t arena_count_override = SIZE_MAX; // Lets a fixture report fewer evictable pages than exist (W4).
bool arena_fail = false, arena_lock_violation = false;
namespace ddo_vmm_detail {
ArenaPressureCount pressure_count_arena_pages(int) noexcept {
    ++arena_count_calls;
    if (!pool_mutex.try_lock()) arena_lock_violation = true; else pool_mutex.unlock();
    return {arena_count_override == SIZE_MAX ? arena_pages : arena_count_override, arena_busy};
}
ArenaEvictionResult pressure_evict_arena_page(int) noexcept {
    ++arena_evict_calls;
    if (!pool_mutex.try_lock()) arena_lock_violation = true; else pool_mutex.unlock();
    ArenaEvictionResult result; result.busy_arenas = arena_busy;
    if (arena_fail) { result.outcome = ArenaEviction::failed; return result; }
    if (!arena_pages) return result;
    --arena_pages; result.outcome = ArenaEviction::evicted; result.bytes = arena_page_bytes;
    return result;
}
#ifdef DDO_VMM_TIMING_COUNTERS
// Composition double: configurable totals; records calls and lock violations.
ArenaComposition fixture_composition{true, 0, 1, 3000, 2, 1500, 0, 0};
size_t composition_calls = 0;
ArenaComposition observe_arena_composition(int) noexcept {
    ++composition_calls;
    if (!pool_mutex.try_lock()) arena_lock_violation = true; else pool_mutex.unlock();
    return fixture_composition;
}
#endif
}
#endif
extern "C" {
size_t ddo_vmm_device_granularity(int) { return fixture_page; }
const char* ddo_vmm_last_error() { return "CPU fixture refusal"; }
ddo_vmm_status ddo_vmm_region_create(int, size_t, ddo_vmm_region** output) {
    ++region_attempts; *output = nullptr; return region_status;
}
void ddo_vmm_region_destroy(ddo_vmm_region*) {}
uintptr_t ddo_vmm_region_address(const ddo_vmm_region*) { return 0; }
size_t ddo_vmm_region_mapped_size(const ddo_vmm_region*) { return 0; }
ddo_vmm_status ddo_vmm_region_evict(ddo_vmm_region*) { return DDO_VMM_INVALID_ARGUMENT; }
int ddo_vmm_pressure_evict_one(int) { return 0; }
ddo_vmm_status ddo_vmm_pressure_prepare_cuda_allocation(int, size_t bytes) {
    ++pressure_calls; last_pressure_bytes = bytes;
#ifdef DDO_VMM_AUTO_SEGMENTED_CANDIDATE
    if (auto_pressure) return auto_pressure(bytes);
#endif
    return pressure_status;
}
}

struct PhysicalDriver {
    using Handle = unsigned;
    std::array<std::array<unsigned char, fixture_page>, 32> bytes{};
    std::array<bool, 32> handles{};
    std::array<Handle, 4> mappings{};
    unsigned next = 1;
    bool lock_violation = false, va_live = false;
    int calls = 0;
    std::string fault;
    std::function<void()> during;
    bool identity(int device) noexcept { return device == 0; }
    int step(const char* operation) noexcept {
        ++calls;
        if (!pool_mutex.try_lock()) { lock_violation = true; return 99; }
        pool_mutex.unlock();
        if (during) { auto callback = std::move(during); callback(); }
        return fault == operation ? 600 : 0;
    }
    int create(size_t, Handle& handle) noexcept {
        if (const int error = step("create")) return error;
        if (next >= handles.size()) return 98;
        handle = next++; handles[handle] = true; bytes[handle].fill(0); return 0;
    }
    int map(uintptr_t address, size_t, Handle handle) noexcept {
        if (const int error = step("map")) return error;
        mappings[(address - fixture_base) / fixture_page] = handle; return 0;
    }
    int access(uintptr_t, size_t) noexcept { return step("access"); }
    int unmap(uintptr_t address, size_t) noexcept {
        if (const int error = step("unmap")) return error;
        mappings[(address - fixture_base) / fixture_page] = 0; return 0;
    }
    int release(Handle handle) noexcept {
        if (const int error = step("release")) return error;
        handles[handle] = false; return 0;
    }
    int reserve(size_t, size_t, uintptr_t& address) noexcept {
        if (const int error = step("reserve")) return error;
        address = fixture_base; va_live = true; return 0;
    }
    int free_va(uintptr_t, size_t) noexcept {
        if (const int error = step("free_va")) return error;
        va_live = false; return 0;
    }
};
struct FixtureAccount {
    size_t va = 0, physical = 0;
    bool claim_va(size_t count) noexcept { if (va || count != 4 * fixture_page) return false; va = count; return true; }
    void credit_va(size_t count) noexcept { va -= count; }
};
struct CpuStorage final : SegmentedPoolStorage {
    PhysicalDriver driver;
    FixtureAccount account;
    ddo_dead_pages::FactoryOwner<PhysicalDriver, FixtureAccount> factory;
    SpanLedger& ledger;
    ddo_dead_pages::Storage<PhysicalDriver>& storage;
    size_t& charge;
    size_t ceiling = 4 * fixture_page;
    int close_error = 0;
    static SpanLedger& open(decltype(factory)& owner) {
        if (!owner.open([]() noexcept { return true; })) throw std::runtime_error("factory open");
        return owner.ledger();
    }
    CpuStorage() : factory(driver, account, 4 * fixture_page, fixture_page, 0, 7),
        ledger(open(factory)), storage(factory.storage()), charge(account.physical) {}
    bool materialize(const SpanLedger::Lease& lease, const SpanLedger& value,
            std::unique_lock<std::mutex>& lock) noexcept override {
        bool result = storage.materialize(lease, value, [&](size_t count) noexcept {
            if (count && ddo_vmm_pressure_prepare_cuda_allocation(0, count) != DDO_VMM_SUCCESS) return false;
            if (count > ceiling - charge) return false;
            charge += count; lock.unlock(); return true;
        });
        if (!lock.owns_lock()) lock.lock();
        return result;
    }
    bool discard(size_t index, const SpanLedger& value, std::unique_lock<std::mutex>& lock) noexcept override {
        bool result = storage.discard(index, value, [&](size_t count) noexcept {
            lock.lock(); charge -= count; lock.unlock();
        }, [&]() noexcept { lock.unlock(); });
        if (!lock.owns_lock()) lock.lock();
        return result;
    }
    bool close_va(std::unique_lock<std::mutex>& lock) noexcept override {
        bool result = storage.close_dead(ledger, [&](size_t count) noexcept {
            lock.lock(); charge -= count; lock.unlock();
        }, [&]() noexcept { lock.unlock(); });
        if (result) { result = factory.close(); close_error = factory.cleanup_error(); }
        if (!lock.owns_lock()) lock.lock();
        return result;
    }
    size_t page_count() const noexcept override { return 4; }
    size_t physical_charge() const noexcept override { return charge; }
    int error() const noexcept override { return storage.error() ? storage.error() : close_error; }
};
struct Fixture {
    std::shared_ptr<Backing> owner;
    std::shared_ptr<CpuStorage> physical;
    Fixture() {
        check(backings.empty() && records.empty(), "clean fixture entry");
        events_complete = false; dependency_failure = query_failure = false; region_attempts = 0;
        region_status = DDO_VMM_OUT_OF_MEMORY;
        pressure_status = DDO_VMM_SUCCESS; pressure_calls = last_pressure_bytes = 0;
        physical = std::make_shared<CpuStorage>();
        owner = std::make_shared<Backing>(physical, physical->ledger, physical->factory.address(), 0, fixture_context());
        backings.push_back(owner);
        check(&owner->ledger == &physical->factory.ledger() && !owner->local_ledger,
              "factory and pool share exactly one ledger");
        // Exclude the factory's VA reservation from operation-injection counters.
        physical->driver.calls = 0;
    }
    void* allocate(size_t count, CUstream stream = fixture_stream()) {
        return ddo_vmm_pool_malloc(count, 0, reinterpret_cast<cudaStream_t>(stream));
    }
    void retire(void* pointer) { ddo_vmm_pool_free(pointer, reinterpret_cast<cudaStream_t>(fixture_stream())); }
    void preserve_failed_fixture() {
        // CPU harness isolation only: retain failed owners in an explicit archive.
        static auto* retained = new std::vector<std::shared_ptr<Backing>>;
        for (auto& item : backings) retained->push_back(item);
        records.clear(); backings.clear();
    }
};
int main() {
    try {
        { Fixture f;
          auto left = f.allocate(fixture_page), middle = f.allocate(2 * fixture_page), right = f.allocate(fixture_page);
          check(left && middle && right && records.size() == 3, "actual pool published records");
          const auto left_handle = f.physical->storage.unit(0).handle;
          const auto right_handle = f.physical->storage.unit(3).handle;
          f.physical->driver.bytes[left_handle].fill(17); f.physical->driver.bytes[right_handle].fill(83);
          check(!close_segmented(f.owner), "live leases block close");
          f.retire(middle);
          check(reclaim_segmented(f.owner).released == 0, "pending original fence blocks reclaim");
          events_complete = true;
          check(reclaim_segmented(f.owner).released == 2, "dead middle units reclaimed");
          check(f.physical->charge == 2 * fixture_page && records.size() == 2, "physical charge distinct from live records");
          auto report = segmented_report();
          check(report.available && report.owners == 1 && report.virtual_bytes == 4 * fixture_page
              && report.physical_charge == 2 * fixture_page && report.live_bytes == 2 * fixture_page,
              "report separates reserved VA and physical charge");
          uint64_t legacy_stats[15]{};
          check(ddo_vmm_pool_stats(legacy_stats, 15) != 0, "do not mislabel VA as physical");
          f.physical->driver.during = [&]() {
              check(!close_segmented(f.owner), "maintenance blocks close");
              check(reclaim_segmented(f.owner).scanned == 0, "maintenance blocks recursive recovery");
              check(f.owner->retiring, "owner retained during driver call");
          };
          auto output = f.allocate(2 * fixture_page);
          check(output == middle && records.size() == 3, "actual pool stable VA reuse");
          check(f.physical->driver.bytes[left_handle][0] == 17
              && f.physical->driver.bytes[right_handle][fixture_page - 1] == 83, "live neighbors exact");
          f.retire(left); f.retire(output); f.retire(right);
          check(close_segmented(f.owner), "checked complete close");
          check(backings.empty() && records.empty() && f.physical->charge == 0
              && !f.physical->account.va && !f.physical->driver.va_live
              && !f.physical->driver.lock_violation, "final ownership zero");
          for (bool live : f.physical->driver.handles) check(!live, "all handles released");
        }
        for (const std::string operation : {"create", "map", "access", "unmap", "release", "free_va"}) {
            Fixture f;
            if (operation == "create" || operation == "map" || operation == "access") {
                f.physical->driver.fault = operation;
                check(!f.allocate(fixture_page) && records.empty(), "failed physical creation publishes nothing");
            } else {
                auto pointer = f.allocate(fixture_page); check(pointer != nullptr, "release fault fixture");
                f.retire(pointer); events_complete = true; f.physical->driver.fault = operation;
                check(!close_segmented(f.owner), "failed release retains owner");
            }
            check(f.physical->error() == 600 && f.owner->retiring && backings.size() == 1, "original failure and owner retained");
            const auto calls = f.physical->driver.calls;
            check(!close_segmented(f.owner) && reclaim_segmented(f.owner).scanned == 0
                && f.physical->driver.calls == calls, "no retry or forced cleanup");
            check(!f.physical->driver.lock_violation, "failed driver operation outside pool lock");
            f.preserve_failed_fixture();
        }
        { Fixture f; f.physical->ceiling = 0;
          check(!f.allocate(fixture_page) && records.empty() && !f.owner->retiring
              && f.owner->ledger.snapshot().reserved_count == 0 && !f.physical->driver.calls, "admission rollback before driver");
          check(close_segmented(f.owner), "empty reserved VA close"); }
        { Fixture f; auto pointer = f.allocate(fixture_page); f.retire(pointer); query_failure = true;
          auto recovery = reclaim_segmented(f.owner);
          check(recovery.released == 0 && recovery.error == static_cast<int>(CUDA_ERROR_ILLEGAL_ADDRESS)
              && f.owner->segmented_error == static_cast<int>(CUDA_ERROR_ILLEGAL_ADDRESS)
              && !close_segmented(f.owner), "failed original fence error propagates and retains mapping");
          check(f.physical->charge == fixture_page && f.physical->driver.va_live, "failed fence no release credit");
          f.preserve_failed_fixture(); query_failure = false; }
        { Fixture f; auto pointer = f.allocate(fixture_page); f.retire(pointer);
          dependency_failure = true;
          auto different_stream = reinterpret_cast<CUstream>(uintptr_t{0x300});
          check(!f.allocate(fixture_page, different_stream) && records.empty()
              && f.owner->ledger.snapshot().quarantined == fixture_page,
              "dependency failure before record publication quarantines lease");
          check(!close_segmented(f.owner) && f.physical->charge == fixture_page, "submission failure retains charge");
          f.preserve_failed_fixture(); dependency_failure = false; }
        { Fixture f; auto pointer = f.allocate(fixture_page);
          check(f.allocate(fixture_page) != nullptr, "live neighbor forces second-pass search");
          f.retire(pointer); query_failure = true;
          check(!f.allocate(4 * fixture_page) && region_attempts == 0
              && f.owner->segmented_error == static_cast<int>(CUDA_ERROR_ILLEGAL_ADDRESS),
              "allocation second-pass query error stops before new region");
          f.preserve_failed_fixture(); query_failure = false; }
        { Fixture f; f.owner->retiring = true;
          std::unique_lock lock(pool_mutex); PoolAllocationAttempt attempt(fixture_page, 0, 0, allocation_failure_sink);
          check(!publish_segmented(f.owner, fixture_page, nullptr, attempt, lock)
              && f.owner->ledger.snapshot().reserved_count == 0, "retiring owner cannot reserve");
          lock.unlock(); f.owner->retiring = false;
          check(close_segmented(f.owner), "untouched retiring fixture closes"); }
        { Fixture f; auto dead = f.allocate(fixture_page); auto live = f.allocate(fixture_page);
          f.retire(dead); events_complete = true;
          f.physical->driver.during = [&]() {
              check(!recover_segmented_context(0, fixture_context()).scanned, "recovery recursion guard");
          };
          check(!f.allocate(4 * fixture_page), "pressure fixture region refusal preserved");
          check(region_attempts == 2 && f.physical->charge == fixture_page
              && f.physical->account.va == 4 * fixture_page && records.size() == 1,
              "one allocation retry only after completed physical release");
          check(!f.physical->storage.unit(0).handle && f.physical->storage.unit(1).handle,
              "pressure recovery excludes live page");
          f.retire(live); check(close_segmented(f.owner), "pressure fixture closes"); }
        { Fixture f; auto dead = f.allocate(fixture_page); auto live = f.allocate(fixture_page);
          f.retire(dead); events_complete = true; region_status = DDO_VMM_CUDA_ERROR;
          check(!f.allocate(4 * fixture_page) && region_attempts == 1
              && f.physical->charge == 2 * fixture_page, "non OOM cannot trigger segmented recovery");
          f.retire(live); check(close_segmented(f.owner), "non OOM fixture closes"); }
        { Fixture f; auto pointer = f.allocate(4 * fixture_page); f.retire(pointer); events_complete = true;
          auto recovery = reclaim_segmented(f.owner, 2, 1);
          check(recovery.scanned == 1 && recovery.released == 1 && f.physical->charge == 3 * fixture_page,
              "caller recovery bounds constrain work");
          check(close_segmented(f.owner), "bounded fixture closes"); }
        { std::unique_lock lock(pool_mutex);
          check(!segmented_report().available, "busy report unavailable not zero"); }
        { Fixture f; pressure_status = DDO_VMM_CUDA_ERROR;
          check(!f.allocate(fixture_page) && pressure_calls == 1 && last_pressure_bytes == fixture_page
              && !f.physical->driver.calls && !f.physical->charge && records.empty()
              && f.owner->ledger.snapshot().reserved_count == 0,
              "native admission refusal before physical mutation and publication");
          check(close_segmented(f.owner), "admission-refused factory closes"); }
        check(segmented_report().available && !segmented_report().owners, "empty final report");
        check(!GetModuleHandleW(L"nvcuda.dll") && !GetModuleHandleW(L"cudart64_13.dll"), "no CUDA module loaded");
        std::cout << "PASS actual pool CPU: " << checks << " checks; CUDA symbols substituted, no GPU\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
