#include "ddo_vmm.h"
#include "ddo_vmm_timeline.h"
#include "ddo_vmm_source_scheduler.h"
#include "ddo_vmm_source_flights.h"
#include "ddo_vmm_pressure_observations.h"
#include "source_io.h"
#include "host_accounting.h"
#include "host_platform.h"
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <thread>

namespace {
thread_local char error_text[512]{};
constexpr uint64_t micro_limit_bytes = 128ULL * 1024 * 1024;
constexpr uint64_t extended_process_limit_bytes = 1024ULL * 1024 * 1024;
constexpr uint64_t model_capacity_limit_bytes = 32ULL * 1024 * 1024 * 1024;
constexpr uint64_t model_process_limit_bytes = 40ULL * 1024 * 1024 * 1024;
auto& aggregate_mutex = ddo_host_accounting::mutex;
std::unordered_map<const void*, uint64_t> aggregate_limits;
auto& aggregate_registered_bytes = ddo_host_accounting::cache_registered_bytes;
auto& aggregate_committed_bytes = ddo_host_accounting::cache_committed_bytes;

// Caller holds aggregate_mutex. The admission limit includes native staging.
uint64_t effective_process_limit() {
    uint64_t limit = model_process_limit_bytes;
    for (const auto& item : aggregate_limits) limit = std::min(limit, item.second);
    return limit;
}
void refresh_ring_guard();
struct AggregateLease {
    const void* key;
    bool active = false;
    ~AggregateLease() {
        if (active) {
            std::lock_guard<std::mutex> lock(aggregate_mutex);
            aggregate_limits.erase(key);
            refresh_ring_guard();
        }
    }
};
ddo_vmm_status failure(ddo_vmm_status status, const char* text) noexcept {
    std::snprintf(error_text, sizeof(error_text), "%s", text);
    return status;
}
template<class F> ddo_vmm_status boundary(F&& f) noexcept {
    error_text[0] = 0;
    try { return f(); }
    catch (const std::bad_alloc&) { return failure(DDO_VMM_OUT_OF_MEMORY, "source cache allocation failed"); }
    catch (const std::exception& e) { return failure(DDO_VMM_INTERNAL_ERROR, e.what()); }
    catch (...) { return failure(DDO_VMM_INTERNAL_ERROR, "unexpected source cache failure"); }
}
ddo_vmm_status cuda_status(CUresult result, const char* action) {
    return result == CUDA_SUCCESS ? DDO_VMM_SUCCESS : failure(DDO_VMM_CUDA_ERROR, action);
}
struct Entry {
    ddo_vmm_source_file* source = nullptr;
    ddo_vmm_source_file_info info{};
    uint64_t offset = 0;
    size_t size = 0;
    size_t slot = 0;
    std::mutex mutex;
    std::atomic<bool> ready{false};
    bool committed = false;
    bool registered = false;
    size_t active_dmas = 0;
    ddo_vmm_status failed = DDO_VMM_SUCCESS;
    std::atomic<bool> dispatched_read{false};
    std::condition_variable read_complete;
    ddo_vmm_detail::SourceReadToken read_token{};
    ~Entry() { ddo_vmm_source_file_close(&source); }
};
struct PrimaryContextLease {
    int device;
    bool retained = false;
    ~PrimaryContextLease() { if (retained) cuDevicePrimaryCtxRelease(device); }
};
}

struct ddo_vmm_source_cache {
    std::mutex mutex;
    std::recursive_mutex batch_mutex;
    std::vector<CUevent> free_events;
    std::vector<ddo_vmm_source_batch*> batches;
    ddo_vmm_host_ring* ring = nullptr;
    CUcontext context = nullptr;
    int device = 0;
    size_t slot_size = 0;
    size_t slot_count = 0;
    uint64_t host_headroom = 0;
    bool needs_clear = false;
    bool pressure_suspended = false;
    bool model_windows = false;
    bool persistent_buffer = false;
    // Source-only dispatcher selected for this cache: requires the dispatcher
    // build and a persistent buffer; may be disabled before the first request.
    bool dispatcher = false;
    uint64_t registered_limit = 0;
    std::vector<ddo_vmm_source_ticket*> transfers;
    std::vector<std::shared_ptr<Entry>> entries;
    ddo_vmm_source_cache_stats stats{};
    ddo_vmm_detail::PressureObservations pressure_observations;
    std::unique_ptr<ddo_vmm_detail::SourceReadScheduler> read_scheduler;
    std::condition_variable read_available;
    std::thread read_dispatcher;
    bool stopping_reads = false;
    size_t read_flights_owned = 0, read_flights_peak = 0, read_flights_limit = 0;
#if defined(DDO_VMM_SOURCE_TEST_HOOKS)
    int64_t copies_before_failure = -1;
    bool fail_close_once = false;
#endif
};
struct ddo_vmm_source_ticket {
    ddo_vmm_source_cache* cache = nullptr;
    std::shared_ptr<Entry> entry;
    size_t relative_offset = 0;
    size_t size = 0;
    CUevent event = nullptr;
    CUstream stream = nullptr;
    bool submitted = false;
    bool event_recorded = false;
    bool registration_lease = false;
};

struct ddo_vmm_source_batch {
    ddo_vmm_source_cache* cache = nullptr;
    std::vector<ddo_vmm_source_span> spans;
    std::vector<ddo_vmm_source_ticket*> pending;
    ddo_vmm_host_ring* ring = nullptr;
    size_t ring_slots = 0;
    size_t ring_slot_bytes = 0;
    size_t next_slot = 0;
    CUevent completion = nullptr;
    bool completion_recorded = false;
    bool unsafe_completion = false;
    ~ddo_vmm_source_batch() {
        if (completion) cuEventDestroy(completion);
        for (auto& span : spans) ddo_vmm_source_file_close(&span.source);
    }
};

namespace {
struct ApiTimer {
    ddo_vmm_source_cache* cache;
    double ddo_vmm_source_cache_stats::* field;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    ~ApiTimer() {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->stats.*field += elapsed;
    }
};

void refresh_ring_guard() {
    ddo_host_accounting::active_cache_limit_bytes = aggregate_limits.empty()
        ? UINT64_MAX : effective_process_limit();
    uint64_t headroom = 0;
    for (const auto& item : aggregate_limits) {
        auto cache = static_cast<const ddo_vmm_source_cache*>(item.first);
        headroom = std::max(headroom, cache->host_headroom);
    }
    ddo_host_accounting::active_cache_headroom_bytes = headroom;
}
ddo_vmm_status context_check(ddo_vmm_source_cache* cache) {
    CUcontext current = nullptr;
    if (!cache) return failure(DDO_VMM_INVALID_ARGUMENT, "cache is closed");
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != cache->context)
        return failure(DDO_VMM_CUDA_ERROR, "cache requires its owning CUDA context to be current");
    return DDO_VMM_SUCCESS;
}
ddo_vmm_status wait_entry(ddo_vmm_source_cache* cache, Entry& entry) {
    std::unique_lock<std::mutex> lock(entry.mutex);
    if (entry.dispatched_read)
        entry.read_complete.wait(lock, [&entry] { return entry.ready.load() || entry.failed != DDO_VMM_SUCCESS; });
    if (entry.failed != DDO_VMM_SUCCESS)
        return failure(entry.failed, "source cache entry failed; clear before retrying");
    if (!entry.ready) {
        auto status = ddo_vmm_host_ring_wait_file_read(cache->ring, entry.slot);
        if (status != DDO_VMM_SUCCESS) {
            entry.failed = status;
            return failure(status, ddo_vmm_last_error());
        }
        entry.ready = true;
    }
    auto info = entry.info;
    auto status = ddo_vmm_source_file_get_info(entry.source, &info);
    if (status != DDO_VMM_SUCCESS) {
        entry.failed = status;
        return failure(status, ddo_vmm_source_last_error());
    }
    return DDO_VMM_SUCCESS;
}
bool same_file(const ddo_vmm_source_file_info& a, const ddo_vmm_source_file_info& b) {
    return a.volume_id == b.volume_id && a.file_id == b.file_id
        && a.size_bytes == b.size_bytes && a.last_write_time == b.last_write_time;
}

// One metadata dispatcher feeds the existing eight chunk-I/O workers. It has
// no CUDA calls/context, no tensor owners and no new host or registration cap.
// The cache owns ring storage until stop_reads() joins this dispatcher.
void dispatch_reads(ddo_vmm_source_cache* cache) noexcept {
#if defined(_WIN32) || defined(__linux__)
    using Work = ddo_vmm_detail::SourceReadScheduler::Work;
    using Flights = ddo_vmm_detail::SourceReadFlights<Work, ddo_vmm_status>;
    // Same existing worker/slot geometry, not a new host/GPU memory ceiling.
    Flights flights(std::min({cache->slot_count, Flights::max_capacity,
                             ddo_vmm_host_ring_source_worker_count(cache->ring)}));
    {
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->read_flights_limit = flights.limit();
    }
    auto retire = [cache](Flights::Flight& flight) {
        auto entry = std::static_pointer_cast<const Entry>(flight.work->read_owner);
        auto drained = ddo_vmm_host_ring_wait_file_read(cache->ring, entry->slot);
        const auto status = flight.submitted == DDO_VMM_SUCCESS ? drained : flight.submitted;
        auto mutable_entry = std::const_pointer_cast<Entry>(entry);
        {
            std::lock_guard<std::mutex> lock(mutable_entry->mutex);
            mutable_entry->failed = status;
            if (status == DDO_VMM_SUCCESS) mutable_entry->ready = true;
        }
        mutable_entry->read_complete.notify_all();
        {
            std::lock_guard<std::mutex> lock(cache->mutex);
            cache->read_scheduler->acknowledge(flight.work->token, status == DDO_VMM_SUCCESS);
            cache->read_scheduler->cancel(flight.work->token);
            --cache->read_flights_owned;
        }
    };
    for (;;) {
        // Reap any completed window, not just the oldest. A later ready demand
        // may be consumed while another window is still reading.
        flights.reap([cache](const auto& flight) {
            auto entry = std::static_pointer_cast<const Entry>(flight.work->read_owner);
            return ddo_source_io::source_read_complete(cache->ring, entry->slot);
        }, retire);
        std::shared_ptr<const Work> work;
        bool stopping = false;
        {
            std::unique_lock<std::mutex> lock(cache->mutex);
            stopping = cache->stopping_reads;
            if (!stopping && !flights.full()) work = cache->read_scheduler->take();
            if (!stopping && !work) {
                if (flights.empty()) {
                    cache->read_available.wait(lock, [cache] {
                        return cache->stopping_reads || cache->read_scheduler->snapshot().queued != 0;
                    });
                } else {
                    // Native CPU completion polling, max eight futures. Avoid
                    // blocking on the oldest read and respond to new demand.
                    cache->read_available.wait_for(lock, std::chrono::milliseconds(1), [&] {
                        return cache->stopping_reads || (!flights.full()
                            && cache->read_scheduler->snapshot().queued != 0);
                    });
                }
            }
        }
        if (stopping) {
            // begin_close cancels queued metadata only. Submitted/partial reads
            // retain owners until this blocking drain; join precedes ring free.
            flights.reap([](const auto&) { return true; }, retire);
            return;
        }
        if (!work) continue;
        // Capacity was checked before take. Fixed-array adoption cannot allocate
        // after the scheduler hands over a reading lease.
        auto* flight = flights.adopt(work, DDO_VMM_INTERNAL_ERROR);
        {
            std::lock_guard<std::mutex> lock(cache->mutex);
            ++cache->read_flights_owned;
            cache->read_flights_peak = std::max(cache->read_flights_peak, cache->read_flights_owned);
        }
        auto entry = std::static_pointer_cast<const Entry>(work->read_owner);
        try {
            flight->submitted = ddo_source_io::read_source_async(cache->ring, entry->slot,
                entry->source, entry->offset, entry->size);
        } catch (...) {
            // Completion was published before chunks. Keep the flight until
            // its drain proves all submitted/abandoned chunks are complete.
            flight->submitted = DDO_VMM_INTERNAL_ERROR;
        }
    }
#else
    (void)cache;
#endif
}

// Caller owns cache->mutex, and storage has already passed existing admission.
[[maybe_unused]] ddo_vmm_status schedule_read(ddo_vmm_source_cache* cache,
    const std::shared_ptr<Entry>& entry, bool demand) {
    using namespace ddo_vmm_detail;
    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (entry->ready || entry->failed != DDO_VMM_SUCCESS) return DDO_VMM_SUCCESS;
    }
    if (!cache->read_scheduler) {
        cache->read_scheduler = std::make_unique<SourceReadScheduler>(
            reinterpret_cast<uintptr_t>(cache), cache->slot_count, cache->slot_count);
        cache->stopping_reads = false;
    }
    if (!cache->read_dispatcher.joinable())
        cache->read_dispatcher = std::thread(dispatch_reads, cache);
    auto result = cache->read_scheduler->request({entry->info.volume_id,
        entry->info.file_id, entry->info.size_bytes, entry->info.last_write_time,
        entry->offset, entry->size}, entry, demand);
    if (result.status == SourceReadAdmission::closed || result.status == SourceReadAdmission::deferred)
        return DDO_VMM_BYPASS;
    entry->read_token = result.token;
    // This bit is published before the worker can take cache->mutex. Consumers
    // wait on the entry CV, not a not-yet-initialized ring future.
    entry->dispatched_read = true;
    cache->read_available.notify_one();
    return DDO_VMM_SUCCESS;
}

void stop_reads(ddo_vmm_source_cache* cache) {
    {
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->stopping_reads = true;
        if (cache->read_scheduler) cache->read_scheduler->begin_close();
    }
    cache->read_available.notify_all();
    if (cache->read_dispatcher.joinable()) cache->read_dispatcher.join();
}

ddo_vmm_status admit_slot(ddo_vmm_source_cache* cache, Entry& entry) {
#if defined(_WIN32) || defined(__linux__)
    // Lock order: cache -> aggregate -> ring. Workers never take the first two.
    std::lock_guard<std::mutex> lock(aggregate_mutex);
    const uint64_t limit = effective_process_limit();
    const uint64_t staging = ddo_host_accounting::staging_registered_bytes;
    ddo_host_platform::Headroom memory{};
    if (staging > limit || aggregate_committed_bytes > limit - staging
        || cache->slot_size > limit - staging - aggregate_committed_bytes
        || !ddo_host_platform::headroom(memory)
        || !ddo_host_platform::permits(memory, cache->slot_size, cache->host_headroom)) {
        ++cache->stats.admission_denials;
        return DDO_VMM_BYPASS;
    }
    auto status = cache->persistent_buffer
        ? ddo_source_io::commit_storage(cache->ring, entry.slot)
        : ddo_source_io::commit_slot(cache->ring, entry.slot);
    if (status == DDO_VMM_OUT_OF_MEMORY) {
        ++cache->stats.admission_denials;
        return DDO_VMM_BYPASS;
    }
    if (status != DDO_VMM_SUCCESS) return failure(status, ddo_vmm_last_error());
    entry.committed = true;
    aggregate_committed_bytes += cache->slot_size;
    cache->stats.committed_bytes += cache->slot_size;
    cache->stats.peak_committed_bytes = std::max(cache->stats.peak_committed_bytes, cache->stats.committed_bytes);
    entry.registered = !cache->persistent_buffer;
    if (entry.registered) {
        aggregate_registered_bytes += cache->slot_size;
        cache->stats.registered_bytes += cache->slot_size;
        ++cache->stats.registrations;
    }
    cache->stats.peak_registered_bytes = std::max(cache->stats.peak_registered_bytes, cache->stats.registered_bytes);
    return DDO_VMM_SUCCESS;
#else
    (void)cache; (void)entry;
    return DDO_VMM_UNSUPPORTED;
#endif
}

uint64_t saturating_add(uint64_t left, uint64_t right) {
    return right > UINT64_MAX - left ? UINT64_MAX : left + right;
}

// Caller holds cache->mutex. Releasing only the tail preserves the dense slot
// index and avoids moving registered bytes while another entry is in use.
ddo_vmm_status unregister_entry(ddo_vmm_source_cache* cache, Entry& entry) {
#if defined(_WIN32) || defined(__linux__)
    if (!entry.registered) return DDO_VMM_SUCCESS;
    if (!entry.ready || entry.active_dmas) return DDO_VMM_BUSY;
    std::lock_guard<std::mutex> aggregate_lock(aggregate_mutex);
    const auto started = std::chrono::steady_clock::now();
    auto status = ddo_source_io::unregister_storage(cache->ring, entry.slot);
    cache->stats.unregistration_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    if (status != DDO_VMM_SUCCESS) return failure(status, ddo_vmm_last_error());
    entry.registered = false;
    aggregate_registered_bytes -= cache->slot_size;
    cache->stats.registered_bytes -= cache->slot_size;
    ++cache->stats.unregistrations;
    return DDO_VMM_SUCCESS;
#else
    (void)cache; (void)entry;
    return DDO_VMM_UNSUPPORTED;
#endif
}

ddo_vmm_status trim_ready_tail(ddo_vmm_source_cache* cache, uint64_t target, bool discard_storage = false) {
#if defined(_WIN32) || defined(__linux__)
    if (cache->persistent_buffer && !discard_storage) {
        for (auto entry = cache->entries.rbegin(); entry != cache->entries.rend() && cache->stats.registered_bytes > target; ++entry) {
            if (!(*entry)->registered || !(*entry)->ready || (*entry)->active_dmas) continue;
            auto status = unregister_entry(cache, **entry);
            if (status != DDO_VMM_SUCCESS) return status;
        }
        return cache->stats.registered_bytes <= target ? DDO_VMM_SUCCESS : DDO_VMM_BUSY;
    }
    while (cache->stats.committed_bytes > target && !cache->entries.empty()) {
        auto& entry = cache->entries.back();
        if (entry.use_count() != 1 || !entry->ready || !entry->committed)
            return DDO_VMM_BUSY;
        std::lock_guard<std::mutex> aggregate_lock(aggregate_mutex);
        auto status = ddo_source_io::release_slot(cache->ring, entry->slot);
        if (status != DDO_VMM_SUCCESS) return failure(status, ddo_vmm_last_error());
        if (entry->registered) {
            aggregate_registered_bytes -= cache->slot_size;
            cache->stats.registered_bytes -= cache->slot_size;
            ++cache->stats.unregistrations;
        }
        aggregate_committed_bytes -= cache->slot_size;
        cache->stats.committed_bytes -= cache->slot_size;
        cache->stats.retained_bytes -= entry->size;
        ++cache->stats.pressure_evictions;
        cache->stats.pressure_released_bytes += cache->slot_size;
        cache->entries.pop_back();
    }
    return cache->stats.committed_bytes <= target ? DDO_VMM_SUCCESS : DDO_VMM_BUSY;
#else
    (void)cache; (void)target; (void)discard_storage;
    return DDO_VMM_UNSUPPORTED;
#endif
}

// The low/high bands prevent a miss from immediately undoing a pressure trim.
// Existing hits remain usable while growth is suspended. Workers and DMAs do
// not need the cache or process accounting lock to finish.
ddo_vmm_status poll_pressure(ddo_vmm_source_cache* cache) {
#if defined(_WIN32) || defined(__linux__)
    ddo_host_platform::Headroom memory{};
    auto& observations = cache->pressure_observations;
    const uint64_t low = saturating_add(cache->host_headroom, cache->slot_size);
    const uint64_t high = saturating_add(low, cache->slot_size);
    const bool was_suspended = cache->pressure_suspended;
    if (!ddo_host_platform::headroom(memory)) {
        cache->pressure_suspended = true;
        observations.sample(false, 0, 0, low, high);
        observations.transition(was_suspended, true, false, 0, 0, low,
            cache->stats.committed_bytes, cache->stats.registered_bytes);
        return DDO_VMM_SUCCESS;
    }
    observations.sample(true, memory.ram, memory.commit, low, high);
    uint64_t available = std::min(memory.ram, memory.commit);
    if (available < low) cache->pressure_suspended = true;
    else if (available >= high) cache->pressure_suspended = false;
    observations.transition(was_suspended, cache->pressure_suspended, true,
        memory.ram, memory.commit, low, cache->stats.committed_bytes, cache->stats.registered_bytes);
    while (cache->pressure_suspended && available < high && cache->stats.committed_bytes) {
        const auto before_bytes = cache->stats.committed_bytes;
        const auto target = cache->stats.committed_bytes - cache->slot_size;
        auto status = trim_ready_tail(cache, target, true);
        observations.reclaimed(before_bytes - cache->stats.committed_bytes);
        if (status == DDO_VMM_BUSY) { observations.blocked(); break; }
        if (status != DDO_VMM_SUCCESS) { observations.failed(); return status; }
        if (!ddo_host_platform::headroom(memory)) {
            observations.sample(false, 0, 0, low, high);
            break;
        }
        observations.sample(true, memory.ram, memory.commit, low, high);
        available = std::min(memory.ram, memory.commit);
    }
    if (available >= high) {
        observations.transition(cache->pressure_suspended, false, true,
            memory.ram, memory.commit, low, cache->stats.committed_bytes, cache->stats.registered_bytes);
        cache->pressure_suspended = false;
    }
    return DDO_VMM_SUCCESS;
#else
    (void)cache;
    return DDO_VMM_SUCCESS;
#endif
}

// Caller holds cache->mutex, never aggregate_mutex while waiting on CUDA.
ddo_vmm_status retire_transfer(ddo_vmm_source_ticket* ticket, bool wait) {
    if (!ticket->registration_lease) return DDO_VMM_SUCCESS;
    CUresult result;
    if (ticket->event_recorded)
        result = wait ? cuEventSynchronize(ticket->event) : cuEventQuery(ticket->event);
    else if (wait)
        result = cuStreamSynchronize(ticket->stream);
    else
        return DDO_VMM_BUSY;
    if (result == CUDA_ERROR_NOT_READY) return DDO_VMM_BUSY;
    if (result != CUDA_SUCCESS) return failure(DDO_VMM_CUDA_ERROR, "persistent source DMA still owns its registration");
    --ticket->entry->active_dmas;
    ticket->registration_lease = false;
    return DDO_VMM_SUCCESS;
}

ddo_vmm_status ensure_registration(ddo_vmm_source_cache* cache, Entry& entry) {
#if defined(_WIN32) || defined(__linux__)
    if (entry.registered) return DDO_VMM_SUCCESS;
    for (auto* ticket : cache->transfers) {
        auto status = retire_transfer(ticket, false);
        if (status != DDO_VMM_SUCCESS && status != DDO_VMM_BUSY) return status;
    }
    while (cache->stats.registered_bytes > cache->registered_limit - cache->slot_size) {
        bool released = false;
        for (auto& candidate : cache->entries) {
            if (!candidate->registered || candidate->active_dmas || !candidate->ready) continue;
            auto status = unregister_entry(cache, *candidate);
            if (status != DDO_VMM_SUCCESS) return status;
            released = true;
            break;
        }
        if (released) continue;
        auto ticket = std::find_if(cache->transfers.begin(), cache->transfers.end(),
            [](auto* candidate) { return candidate->registration_lease; });
        if (ticket == cache->transfers.end()) return failure(DDO_VMM_BUSY, "no reusable source registration");
        ++cache->stats.registration_waits;
        const auto started = std::chrono::steady_clock::now();
        auto status = retire_transfer(*ticket, true);
        cache->stats.registration_wait_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        if (status != DDO_VMM_SUCCESS) return status;
    }
    std::lock_guard<std::mutex> aggregate_lock(aggregate_mutex);
    const auto started = std::chrono::steady_clock::now();
    auto status = ddo_source_io::register_storage(cache->ring, entry.slot);
    cache->stats.registration_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    if (status == DDO_VMM_OUT_OF_MEMORY) {
        ++cache->stats.admission_denials;
        return DDO_VMM_BYPASS;
    }
    if (status != DDO_VMM_SUCCESS) return failure(status, ddo_vmm_last_error());
    entry.registered = true;
    aggregate_registered_bytes += cache->slot_size;
    cache->stats.registered_bytes += cache->slot_size;
    ++cache->stats.registrations;
    cache->stats.peak_registered_bytes = std::max(cache->stats.peak_registered_bytes, cache->stats.registered_bytes);
    return DDO_VMM_SUCCESS;
#else
    (void)cache; (void)entry;
    return DDO_VMM_UNSUPPORTED;
#endif
}
}

extern "C" const char* ddo_vmm_source_cache_last_error(void) { return error_text; }
extern "C" uint32_t ddo_vmm_source_cache_abi_version(void) { return 11; }

extern "C" ddo_vmm_status ddo_vmm_source_cache_create(
    const ddo_vmm_source_cache_config* config, ddo_vmm_source_cache** output
) {
    return boundary([&]() -> ddo_vmm_status {
        if (!output) return failure(DDO_VMM_INVALID_ARGUMENT, "cache output is required");
        *output = nullptr;
        if (!config || config->version < 1 || config->version > 5
            || config->struct_size < offsetof(ddo_vmm_source_cache_config, process_limit_bytes)
            || (config->version >= 2 && config->struct_size < offsetof(ddo_vmm_source_cache_config, registered_limit_bytes))
            || (config->version == 5 && config->struct_size < sizeof(*config))
            || !config->slot_count || !config->slot_bytes || config->slot_bytes > 64ULL * 1024 * 1024
            || config->slot_bytes % 65536
            || config->slot_count > (config->version >= 4 ? model_capacity_limit_bytes : micro_limit_bytes) / config->slot_bytes)
            return failure(DDO_VMM_INVALID_ARGUMENT, "cache requires aligned slots and capacity within its config version ceiling");
        const uint64_t process_limit = config->version >= 2 ? config->process_limit_bytes : micro_limit_bytes;
        const uint64_t maximum_process_limit = config->version >= 4 ? model_process_limit_bytes
            : config->version == 3 ? extended_process_limit_bytes : micro_limit_bytes;
        if (!process_limit || process_limit > maximum_process_limit)
            return failure(DDO_VMM_INVALID_ARGUMENT, "process cache limit exceeds its config version ceiling");
#if !defined(_WIN32) && !defined(__linux__)
        return failure(DDO_VMM_UNSUPPORTED, "source cache is currently Windows-only");
#else
        const uint64_t capacity = config->slot_count * config->slot_bytes;
        const uint64_t registered_limit = config->version == 5 && config->registered_limit_bytes
            ? config->registered_limit_bytes : capacity;
        if (registered_limit < config->slot_bytes || registered_limit > capacity || registered_limit % config->slot_bytes)
            return failure(DDO_VMM_INVALID_ARGUMENT, "registration ceiling must cover whole segments within storage capacity");
        ddo_host_platform::Headroom memory{};
        if (!ddo_host_platform::headroom(memory))
            return failure(DDO_VMM_BYPASS, "source cache host/cgroup headroom observation unavailable");
        if (!ddo_host_platform::permits(memory, config->slot_bytes, config->host_headroom_bytes))
            return failure(DDO_VMM_BYPASS, "source cache host/cgroup RAM/commit headroom denied");
        auto cache = std::make_unique<ddo_vmm_source_cache>();
        cache->device = config->device_ordinal;
        cache->slot_size = static_cast<size_t>(config->slot_bytes);
        cache->slot_count = config->slot_count;
        cache->host_headroom = config->host_headroom_bytes;
        cache->model_windows = config->version >= 4;
        cache->persistent_buffer = config->version == 5;
        cache->dispatcher = cache->persistent_buffer && ddo_vmm_source_dispatcher_enabled() != 0;
        cache->registered_limit = registered_limit;
        cache->entries.reserve(config->slot_count);
        AggregateLease aggregate_lease{cache.get()};
        {
            std::lock_guard<std::mutex> lock(aggregate_mutex);
            const auto limit = std::min(process_limit, effective_process_limit());
            const auto staging = ddo_host_accounting::staging_registered_bytes;
            if (staging > limit || aggregate_committed_bytes > limit - staging)
                return failure(DDO_VMM_BYPASS, "requested process cache cap is below live registrations");
            aggregate_limits.emplace(cache.get(), process_limit);
            refresh_ring_guard();
            aggregate_lease.active = true;
        }
        CUcontext current = nullptr;
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || !current)
            return failure(DDO_VMM_CUDA_ERROR, "initialize the CUDA primary context before the cache");
        PrimaryContextLease lease{cache->device};
        if (cuDevicePrimaryCtxRetain(&cache->context, cache->device) != CUDA_SUCCESS)
            return failure(DDO_VMM_CUDA_ERROR, "could not retain CUDA primary context");
        lease.retained = true;
        if (cache->context != current) {
            return failure(DDO_VMM_INVALID_ARGUMENT, "cache requires the device primary context");
        }
        auto status = ddo_source_io::create_lazy_ring(cache->device, cache->slot_size, cache->slot_count, &cache->ring);
        if (status != DDO_VMM_SUCCESS) {
            return failure(status, ddo_vmm_last_error());
        }
        cache->stats.struct_size = sizeof(cache->stats);
        cache->stats.version = 8;
        cache->stats.capacity_bytes = capacity;
        cache->stats.registered_limit_bytes = registered_limit;
        lease.retained = false;  // Ownership passes to the successful cache.
        aggregate_lease.active = false;
        *output = cache.release();
        return DDO_VMM_SUCCESS;
#endif
    });
}

static ddo_vmm_status cache_request_impl(
    ddo_vmm_source_cache* cache, const ddo_vmm_source_file* source,
    uint64_t offset, size_t size, ddo_vmm_source_ticket** output, bool demand
) {
#if !defined(DDO_VMM_NATIVE_SOURCE_DISPATCHER)
    (void)demand;
#endif
    return boundary([&]() -> ddo_vmm_status {
        if (!output) return failure(DDO_VMM_INVALID_ARGUMENT, "ticket output is required");
        *output = nullptr;
        if (!cache) return failure(DDO_VMM_INVALID_ARGUMENT, "live cache required");
        ApiTimer timer{cache, &ddo_vmm_source_cache_stats::request_total_seconds};
        auto status = context_check(cache);
        if (status != DDO_VMM_SUCCESS) return status;
        ddo_vmm_source_file_info info{};
        info.struct_size = sizeof(info); info.version = 1;
        status = ddo_vmm_source_file_get_info(source, &info);
        if (status != DDO_VMM_SUCCESS) return failure(status, ddo_vmm_source_last_error());
        if (!size || offset > info.size_bytes || size > info.size_bytes - offset)
            return failure(DDO_VMM_INVALID_ARGUMENT, "cache range exceeds immutable source");
        // Version 4 shares fixed file windows across neighbouring parameters.
        // Normalisation belongs here, not in a Python cache-admission policy.
        // Requests crossing a window bypass; transport splits large tensors.
        const uint64_t fill_offset = cache->model_windows
            ? offset - offset % cache->slot_size : offset;
        const size_t fill_size = cache->model_windows
            ? static_cast<size_t>(std::min<uint64_t>(cache->slot_size, info.size_bytes - fill_offset)) : size;
        std::lock_guard<std::mutex> lock(cache->mutex);
        if (cache->needs_clear) return failure(DDO_VMM_BUSY, "cache cleanup must finish before new requests");
        status = poll_pressure(cache);
        if (status != DDO_VMM_SUCCESS) return status;
        std::shared_ptr<Entry> found;
        bool overlap = false;
        for (auto& entry : cache->entries) {
            if (!same_file(info, entry->info)) continue;
            if (offset >= entry->offset && offset - entry->offset <= entry->size
                && size <= entry->size - (offset - entry->offset)) { found = entry; break; }
            if (offset < entry->offset + entry->size && entry->offset < offset + size) overlap = true;
        }
        auto ticket = std::make_unique<ddo_vmm_source_ticket>();
        if (!found) {
            ++cache->stats.misses;
            if (overlap || size > cache->slot_size || size > fill_size - (offset - fill_offset)
                || cache->entries.size() == cache->slot_count) {
                ++cache->stats.bypasses;
                return DDO_VMM_BYPASS;
            }
            if (cache->pressure_suspended) {
                ++cache->stats.bypasses;
                ++cache->stats.pressure_blocked_requests;
                return DDO_VMM_BYPASS;
            }
            found = std::make_shared<Entry>();
            found->info = info; found->offset = fill_offset; found->size = fill_size;
            found->slot = cache->entries.size();
            status = ddo_vmm_source_file_retain(source, &found->source);
            if (status != DDO_VMM_SUCCESS) return failure(status, ddo_vmm_source_last_error());
            status = admit_slot(cache, *found);
            if (status == DDO_VMM_BYPASS) { ++cache->stats.bypasses; return status; }
            if (status != DDO_VMM_SUCCESS) return status;
            // Publish ownership before scheduling, so partial enqueue failure
            // cannot close the source while a worker still holds its HANDLE.
            cache->entries.push_back(found);
            cache->stats.retained_bytes += fill_size;
#if defined(_WIN32) || defined(__linux__)
#if defined(DDO_VMM_NATIVE_SOURCE_DISPATCHER)
            if (cache->dispatcher) status = schedule_read(cache, found, demand);
            else
#endif
            status = ddo_source_io::read_source_async(cache->ring, found->slot, found->source, fill_offset, fill_size);
            if (status == DDO_VMM_BYPASS) { ++cache->stats.bypasses; return status; }
            if (status != DDO_VMM_SUCCESS) {
                found->failed = status;
                return failure(status, ddo_vmm_last_error());
            }
#endif
        } else {
            ++cache->stats.hits;
#if defined(DDO_VMM_NATIVE_SOURCE_DISPATCHER)
            if (cache->dispatcher) {
                status = schedule_read(cache, found, demand);
                if (status != DDO_VMM_SUCCESS) { ++cache->stats.bypasses; return status; }
            }
#endif
        }
        ticket->cache = cache; ticket->entry = found;
        ticket->relative_offset = static_cast<size_t>(offset - found->offset);
        ticket->size = size;
        ++cache->stats.live_tickets;
        *output = ticket.release();
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_cache_request(
    ddo_vmm_source_cache* cache, const ddo_vmm_source_file* source,
    uint64_t offset, size_t size, ddo_vmm_source_ticket** output
) {
    return cache_request_impl(cache, source, offset, size, output, true);
}

extern "C" int ddo_vmm_source_dispatcher_enabled(void) {
#if defined(DDO_VMM_NATIVE_SOURCE_DISPATCHER) && (defined(_WIN32) || defined(__linux__))
    return 1;
#else
    return 0;
#endif
}

extern "C" ddo_vmm_status ddo_vmm_source_cache_select_dispatcher(ddo_vmm_source_cache* cache, int enabled) {
    return boundary([&]() -> ddo_vmm_status {
        if (!cache || (enabled != 0 && enabled != 1))
            return failure(DDO_VMM_INVALID_ARGUMENT, "live cache and a 0/1 dispatcher selection required");
        std::lock_guard<std::mutex> lock(cache->mutex);
        // Cold selection only: no entry, scheduler or dispatcher thread may exist.
        if (!cache->entries.empty() || cache->read_scheduler || cache->read_dispatcher.joinable())
            return failure(DDO_VMM_BUSY, "dispatcher selection must precede the first cache request");
        if (enabled && !(cache->persistent_buffer && ddo_vmm_source_dispatcher_enabled()))
            return failure(DDO_VMM_UNSUPPORTED, "source-only dispatcher requires its build and a persistent buffer");
        cache->dispatcher = enabled != 0;
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_dispatcher_snapshot(
    ddo_vmm_source_cache* cache, size_t* owned, size_t* peak, size_t* limit) {
    return boundary([&]() -> ddo_vmm_status {
        if (!cache || !owned || !peak || !limit)
            return failure(DDO_VMM_INVALID_ARGUMENT, "live cache and dispatcher snapshot outputs required");
        std::lock_guard<std::mutex> lock(cache->mutex);
        *owned = cache->read_flights_owned;
        *peak = cache->read_flights_peak;
        *limit = cache->read_flights_limit;
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_batch_prepare(ddo_vmm_source_batch* batch) {
    return boundary([&]() -> ddo_vmm_status {
        if (!batch) return failure(DDO_VMM_INVALID_ARGUMENT, "live immutable batch required");
        if (!batch->cache->dispatcher)
            return failure(DDO_VMM_UNSUPPORTED, "source-only dispatcher is not enabled");
        auto* cache = batch->cache;
        auto status = context_check(cache);
        if (status != DDO_VMM_SUCCESS) return status;
        std::lock_guard<std::recursive_mutex> lock(cache->batch_mutex);
        // Metadata and host admission only. Read/window scheduling and lifetime
        // stay native; no arena destination, GPU pin, registration or DMA here.
        size_t attempted_windows = 0;
        for (const auto& span : batch->spans) {
            size_t consumed = 0;
            while (consumed < span.size) {
                // Preparation is optional and bounded by existing geometry,
                // even for huge/repeated descriptors. Demand handles the rest.
                if (attempted_windows++ == cache->slot_count) return DDO_VMM_SUCCESS;
                const auto offset = span.file_offset + consumed;
                const auto bytes = std::min(span.size - consumed,
                    cache->slot_size - static_cast<size_t>(offset % cache->slot_size));
                ddo_vmm_source_ticket* ticket = nullptr;
                status = cache_request_impl(cache, span.source, offset, bytes, &ticket, false);
                if (status == DDO_VMM_BYPASS) return DDO_VMM_SUCCESS;
                if (status != DDO_VMM_SUCCESS && status != DDO_VMM_BYPASS) return status;
                if (ticket) {
                    status = ddo_vmm_source_ticket_close(&ticket);
                    if (status != DDO_VMM_SUCCESS) return status;
                }
                consumed += bytes;
            }
        }
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_ticket_wait(ddo_vmm_source_ticket* ticket) {
    return boundary([&]() -> ddo_vmm_status {
        if (!ticket) return failure(DDO_VMM_INVALID_ARGUMENT, "ticket is closed");
        return wait_entry(ticket->cache, *ticket->entry);
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_ticket_copy_async(
    ddo_vmm_source_ticket* ticket, ddo_vmm_model_arena* arena, size_t offset, void* stream
) {
    return boundary([&]() -> ddo_vmm_status {
        if (!ticket || !arena || ticket->submitted)
            return failure(DDO_VMM_INVALID_ARGUMENT, "copy requires a live, unused ticket and arena");
        if (ddo_vmm_model_arena_device_ordinal(arena) != ticket->cache->device)
            return failure(DDO_VMM_INVALID_ARGUMENT, "cache and arena must belong to the same device");
        ApiTimer timer{ticket->cache, &ddo_vmm_source_cache_stats::copy_total_seconds};
        auto status = context_check(ticket->cache);
        if (status != DDO_VMM_SUCCESS) return status;
        const auto source_started = std::chrono::steady_clock::now();
        status = wait_entry(ticket->cache, *ticket->entry);
        const auto source_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - source_started).count();
        if (status != DDO_VMM_SUCCESS) return status;
        // Publish the lease and its completion event atomically with respect to
        // registration recycling. A recycler must never drain a stream before
        // this ticket has actually enqueued its transfer.
        std::lock_guard<std::mutex> lock(ticket->cache->mutex);
        ticket->cache->stats.source_ready_seconds += source_seconds;
#if defined(DDO_VMM_SOURCE_TEST_HOOKS)
        if (ticket->cache->copies_before_failure == 0) {
            ticket->cache->copies_before_failure = -1;
            return failure(DDO_VMM_INTERNAL_ERROR, "injected copy refusal; prior DMA owners retained");
        }
        if (ticket->cache->copies_before_failure > 0) --ticket->cache->copies_before_failure;
#endif
        const auto event_acquire_started = std::chrono::steady_clock::now();
        if (!ticket->event) {
            if (!ticket->cache->free_events.empty()) {
                ticket->event = ticket->cache->free_events.back();
                ticket->cache->free_events.pop_back();
                ++ticket->cache->stats.events_reused;
            } else {
                auto result = cuEventCreate(&ticket->event, CU_EVENT_DISABLE_TIMING);
                if (result != CUDA_SUCCESS) {
                    ticket->cache->stats.event_acquire_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - event_acquire_started).count();
                    return failure(DDO_VMM_CUDA_ERROR, "cache DMA event creation failed");
                }
                ++ticket->cache->stats.events_created;
            }
        }
        ticket->cache->stats.event_acquire_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - event_acquire_started).count();
        const auto registration_started = std::chrono::steady_clock::now();
        status = ensure_registration(ticket->cache, *ticket->entry);
        ticket->cache->stats.registration_schedule_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - registration_started).count();
        if (status != DDO_VMM_SUCCESS) return status;
        ticket->cache->transfers.push_back(ticket);
        ++ticket->entry->active_dmas;
        ticket->registration_lease = true;
        auto address = ddo_vmm_host_ring_slot_address(ticket->cache->ring, ticket->entry->slot);
        ticket->stream = static_cast<CUstream>(stream);
        ticket->submitted = true;
        const auto submit_started = std::chrono::steady_clock::now();
        status = ddo_vmm_model_arena_copy_from_host_async(arena, offset, ticket->size,
            reinterpret_cast<void*>(address + ticket->relative_offset), stream);
        ticket->cache->stats.dma_submit_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - submit_started).count();
        if (status != DDO_VMM_SUCCESS) return failure(status, ddo_vmm_last_error());
        // If recording fails, the ticket still protects the bytes. Close must
        // drain its stream before releasing that protection.
        const auto event_started = std::chrono::steady_clock::now();
        auto result = cuEventRecord(ticket->event, ticket->stream);
        ticket->cache->stats.dma_event_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - event_started).count();
        if (result != CUDA_SUCCESS) return failure(DDO_VMM_CUDA_ERROR, "cache DMA event recording failed");
        ticket->event_recorded = true;
        ticket->cache->stats.dma_bytes += ticket->size;
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_ticket_wait_on_stream(ddo_vmm_source_ticket* ticket, void* stream) {
    return boundary([&]() -> ddo_vmm_status {
        if (!ticket || !ticket->event_recorded)
            return failure(DDO_VMM_INVALID_ARGUMENT, "ticket has no completed enqueue event");
        auto status = context_check(ticket->cache);
        if (status != DDO_VMM_SUCCESS) return status;
        return cuda_status(cuStreamWaitEvent(static_cast<CUstream>(stream), ticket->event, 0), "cache stream wait failed");
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_ticket_copy_via_ring_async(
    ddo_vmm_source_ticket* ticket, ddo_vmm_host_ring* ring, size_t slot,
    ddo_vmm_model_arena* arena, size_t offset, void* stream
) {
    return boundary([&]() -> ddo_vmm_status {
        if (!ticket || !ring || !arena || ticket->submitted
            || ddo_vmm_model_arena_device_ordinal(arena) != ticket->cache->device)
            return failure(DDO_VMM_INVALID_ARGUMENT, "retained-source fallback requires live owners on the same device");
        auto status = context_check(ticket->cache);
        if (status != DDO_VMM_SUCCESS) return status;
        status = wait_entry(ticket->cache, *ticket->entry);
        if (status != DDO_VMM_SUCCESS) return status;
#if defined(_WIN32) || defined(__linux__)
        auto address = ddo_vmm_host_ring_slot_address(ticket->cache->ring, ticket->entry->slot);
        const auto slot_size = ddo_vmm_host_ring_slot_size(ring);
        if (!slot_size) return failure(DDO_VMM_INVALID_ARGUMENT, "fallback ring has no storage");
        if (!ticket->event && cuEventCreate(&ticket->event, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS)
            return failure(DDO_VMM_CUDA_ERROR, "retained-source fallback event creation failed");
        ticket->stream = static_cast<CUstream>(stream);
        for (size_t copied = 0; copied < ticket->size;) {
            const auto size = std::min(slot_size, ticket->size - copied);
            status = ddo_vmm_host_ring_wait_slot(ring, slot);
            if (status != DDO_VMM_SUCCESS) return failure(status, ddo_vmm_last_error());
            status = ddo_source_io::stage_host_bytes(ring, slot,
                reinterpret_cast<void*>(address + ticket->relative_offset + copied), size);
            if (status != DDO_VMM_SUCCESS) return failure(status, ddo_vmm_last_error());
            ticket->submitted = true;
            status = ddo_vmm_host_ring_copy_to_arena_async(ring, slot, 0, arena, offset + copied, size, stream);
            if (status != DDO_VMM_SUCCESS) return failure(status, ddo_vmm_last_error());
            copied += size;
        }
        if (cuEventRecord(ticket->event, ticket->stream) != CUDA_SUCCESS)
            return failure(DDO_VMM_CUDA_ERROR, "retained-source fallback event recording failed");
        ticket->event_recorded = true;
        std::lock_guard<std::mutex> lock(ticket->cache->mutex);
        ticket->cache->stats.ring_fallback_bytes += ticket->size;
        ticket->cache->stats.host_copy_bytes += ticket->size;
        ticket->cache->stats.dma_bytes += ticket->size;
        return DDO_VMM_SUCCESS;
#else
        (void)slot; (void)offset; (void)stream;
        return DDO_VMM_UNSUPPORTED;
#endif
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_ticket_dma_ready(
    ddo_vmm_source_ticket* ticket, int* output
) {
    return boundary([&]() -> ddo_vmm_status {
        if (!ticket || !output || !ticket->event_recorded)
            return failure(DDO_VMM_INVALID_ARGUMENT, "DMA query requires a ticket with a recorded upload event");
        *output = 0;
        auto status = context_check(ticket->cache);
        if (status != DDO_VMM_SUCCESS) return status;
        const auto result = cuEventQuery(ticket->event);
        if (result == CUDA_SUCCESS) *output = 1;
        else if (result != CUDA_ERROR_NOT_READY)
            return failure(DDO_VMM_CUDA_ERROR, "cache DMA completion query failed");
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_ticket_close(ddo_vmm_source_ticket** owner) {
    return boundary([&]() -> ddo_vmm_status {
        if (!owner) return failure(DDO_VMM_INVALID_ARGUMENT, "ticket owner is required");
        auto* ticket = *owner;
        if (!ticket) return DDO_VMM_SUCCESS;
        ApiTimer timer{ticket->cache, &ddo_vmm_source_cache_stats::close_total_seconds};
        auto status = context_check(ticket->cache);
        if (status != DDO_VMM_SUCCESS) return status;
        // Recycling inspects these ticket events while holding the same lock.
        // Keep the ticket visible and its event alive until its DMA is drained.
        std::lock_guard<std::mutex> lock(ticket->cache->mutex);
#if defined(DDO_VMM_SOURCE_TEST_HOOKS)
        if (ticket->cache->fail_close_once) {
            ticket->cache->fail_close_once = false;
            return failure(DDO_VMM_INTERNAL_ERROR, "injected ticket close refusal; ownership retained");
        }
#endif
        if (ticket->submitted) {
            const auto wait_started = std::chrono::steady_clock::now();
            const auto result = ticket->event_recorded ? cuEventSynchronize(ticket->event) : cuStreamSynchronize(ticket->stream);
            ticket->cache->stats.ticket_close_wait_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - wait_started).count();
            if (result != CUDA_SUCCESS) return failure(DDO_VMM_CUDA_ERROR, "cache ticket still protects a failed DMA");
        }
        const auto event_retire_started = std::chrono::steady_clock::now();
        if (ticket->event) {
            if (ticket->cache->free_events.size() < 256)
                ticket->cache->free_events.push_back(ticket->event);
            else if (cuEventDestroy(ticket->event) != CUDA_SUCCESS)
                return failure(DDO_VMM_CUDA_ERROR, "cache DMA event destroy failed");
            ticket->event = nullptr;
        }
        ticket->cache->stats.event_retire_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - event_retire_started).count();
        if (ticket->registration_lease) {
            --ticket->entry->active_dmas;
            ticket->registration_lease = false;
        }
        auto& transfers = ticket->cache->transfers;
        transfers.erase(std::remove(transfers.begin(), transfers.end(), ticket), transfers.end());
        --ticket->cache->stats.live_tickets;
        delete ticket; *owner = nullptr;
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_cache_clear(ddo_vmm_source_cache* cache) {
    return boundary([&]() -> ddo_vmm_status {
        auto status = context_check(cache);
        if (status != DDO_VMM_SUCCESS) return status;
        {
            std::lock_guard<std::mutex> lock(cache->mutex);
            if (cache->stats.live_tickets) return failure(DDO_VMM_BUSY, "live tickets still own cache bytes");
            cache->needs_clear = true;
        }
        // Never join while holding the lock used for dispatcher acknowledgement.
        // Storage/files/context remain cache-owned until all active chunks drain.
        stop_reads(cache);
        std::lock_guard<std::mutex> lock(cache->mutex);
        // Workers need only ring/source locks, never the cache lock. Retained
        // source handles remain alive until every pending read has drained.
        for (auto& entry : cache->entries) {
            ddo_vmm_host_ring_wait_file_read(cache->ring, entry->slot);
#if defined(_WIN32) || defined(__linux__)
            if (entry->committed) {
                std::lock_guard<std::mutex> aggregate_lock(aggregate_mutex);
                status = ddo_source_io::release_slot(cache->ring, entry->slot);
                if (status != DDO_VMM_SUCCESS) return failure(status, ddo_vmm_last_error());
                if (entry->registered) {
                    aggregate_registered_bytes -= cache->slot_size;
                    cache->stats.registered_bytes -= cache->slot_size;
                    entry->registered = false;
                    ++cache->stats.unregistrations;
                }
                aggregate_committed_bytes -= cache->slot_size;
                cache->stats.committed_bytes -= cache->slot_size;
                entry->committed = false;
            }
#endif
        }
        cache->entries.clear();
        cache->read_scheduler.reset();
        cache->read_flights_owned = cache->read_flights_peak = cache->read_flights_limit = 0;
        cache->stopping_reads = false;
        cache->stats.retained_bytes = 0;
        cache->needs_clear = false;
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_cache_trim(
    ddo_vmm_source_cache* cache, uint64_t target_registered_bytes
) {
    return boundary([&]() -> ddo_vmm_status {
        auto status = context_check(cache);
        if (status != DDO_VMM_SUCCESS) return status;
        std::lock_guard<std::mutex> lock(cache->mutex);
        if (cache->needs_clear) return failure(DDO_VMM_BUSY, "cache cleanup must finish before trim");
        if (cache->persistent_buffer) {
            for (auto* ticket : cache->transfers) {
                status = retire_transfer(ticket, false);
                if (status != DDO_VMM_SUCCESS && status != DDO_VMM_BUSY) return status;
            }
        }
        status = trim_ready_tail(cache, target_registered_bytes);
        return status == DDO_VMM_BUSY ? failure(status, "live or filling tail entry prevents trim") : status;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_cache_pressure_snapshot_v1(
    ddo_vmm_source_cache* cache, uint64_t* values, size_t count) {
    return boundary([&]() -> ddo_vmm_status {
        if (!cache || !values || count != ddo_vmm_detail::PressureObservations::count)
            return failure(DDO_VMM_INVALID_ARGUMENT, "pressure snapshot v1 requires live cache and 23 values");
        std::lock_guard<std::mutex> lock(cache->mutex);
        std::copy(cache->pressure_observations.values.begin(),
                  cache->pressure_observations.values.end(), values);
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_cache_snapshot(
    ddo_vmm_source_cache* cache, ddo_vmm_source_cache_stats* stats
) {
    return boundary([&]() -> ddo_vmm_status {
        if (!cache || !stats || (stats->version < 1 || stats->version > 8)
            || stats->struct_size < offsetof(ddo_vmm_source_cache_stats, capacity_bytes)
            || (stats->version == 2 && stats->struct_size < offsetof(ddo_vmm_source_cache_stats, process_staging_registered_bytes))
            || (stats->version == 3 && stats->struct_size < offsetof(ddo_vmm_source_cache_stats, pressure_evictions))
            || (stats->version == 4 && stats->struct_size < offsetof(ddo_vmm_source_cache_stats, committed_bytes))
            || (stats->version == 5 && stats->struct_size < offsetof(ddo_vmm_source_cache_stats, registration_seconds))
            || (stats->version == 6 && stats->struct_size < offsetof(ddo_vmm_source_cache_stats, source_ready_seconds))
            || (stats->version == 7 && stats->struct_size < offsetof(ddo_vmm_source_cache_stats, request_total_seconds))
            || (stats->version == 8 && stats->struct_size < sizeof(*stats)))
            return failure(DDO_VMM_INVALID_ARGUMENT, "snapshot requires compatible stats ABI");
        std::lock_guard<std::mutex> lock(cache->mutex);
        auto result = cache->stats;
        result.version = stats->version;
        result.struct_size = stats->version == 1 ? offsetof(ddo_vmm_source_cache_stats, capacity_bytes)
            : stats->version == 2 ? offsetof(ddo_vmm_source_cache_stats, process_staging_registered_bytes)
            : stats->version == 3 ? offsetof(ddo_vmm_source_cache_stats, pressure_evictions)
            : stats->version == 4 ? offsetof(ddo_vmm_source_cache_stats, committed_bytes)
            : stats->version == 5 ? offsetof(ddo_vmm_source_cache_stats, registration_seconds)
            : stats->version == 6 ? offsetof(ddo_vmm_source_cache_stats, source_ready_seconds)
            : stats->version == 7 ? offsetof(ddo_vmm_source_cache_stats, request_total_seconds) : sizeof(result);
        result.pressure_suspended = cache->pressure_suspended ? 1 : 0;
        result.source_read_bytes = ddo_vmm_host_ring_source_read_bytes(cache->ring);
        {
            std::lock_guard<std::mutex> aggregate_lock(aggregate_mutex);
            result.process_registered_bytes = aggregate_registered_bytes;
            result.process_committed_bytes = aggregate_committed_bytes;
            result.process_limit_bytes = effective_process_limit();
            result.process_staging_registered_bytes = ddo_host_accounting::staging_registered_bytes;
        }
        std::memcpy(stats, &result, result.struct_size);
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_owned_events_v1(
    ddo_vmm_source_cache* cache, size_t* transfers, size_t* pooled, size_t* batches
) {
    return boundary([&]() -> ddo_vmm_status {
        if (!cache || !transfers || !pooled || !batches)
            return failure(DDO_VMM_INVALID_ARGUMENT, "live cache and three event outputs required");
        std::lock_guard<std::recursive_mutex> batch_lock(cache->batch_mutex);
        std::lock_guard<std::mutex> lock(cache->mutex);
        size_t transfer_count = 0, batch_count = 0;
        for (const auto* ticket : cache->transfers) if (ticket->event) ++transfer_count;
        for (const auto* batch : cache->batches) if (batch->completion) ++batch_count;
        *transfers = transfer_count; *pooled = cache->free_events.size(); *batches = batch_count;
        return DDO_VMM_SUCCESS;
    });
}

#if defined(DDO_VMM_SOURCE_TEST_HOOKS)
extern "C" ddo_vmm_status ddo_vmm_source_test_failures(
    ddo_vmm_source_cache* cache, int64_t copies_before_failure, int close_once
) {
    return boundary([&]() -> ddo_vmm_status {
        if (!cache || copies_before_failure < -1 || (close_once != 0 && close_once != 1))
            return failure(DDO_VMM_INVALID_ARGUMENT, "invalid isolated source failure configuration");
        std::lock_guard<std::mutex> lock(cache->mutex);
        cache->copies_before_failure = copies_before_failure;
        cache->fail_close_once = close_once != 0;
        return DDO_VMM_SUCCESS;
    });
}
#endif

extern "C" ddo_vmm_status ddo_vmm_source_cache_close(ddo_vmm_source_cache** owner) {
    return boundary([&]() -> ddo_vmm_status {
        if (!owner) return failure(DDO_VMM_INVALID_ARGUMENT, "cache owner is required");
        auto* cache = *owner;
        if (!cache) return DDO_VMM_SUCCESS;
        {
            std::lock_guard<std::recursive_mutex> batches_lock(cache->batch_mutex);
            if (!cache->batches.empty()) return failure(DDO_VMM_BUSY, "immutable source batches still own the cache");
        }
        auto status = ddo_vmm_source_cache_clear(cache);
        if (status != DDO_VMM_SUCCESS) return status;
        while (!cache->free_events.empty()) {
            if (cuEventDestroy(cache->free_events.back()) != CUDA_SUCCESS)
                return failure(DDO_VMM_CUDA_ERROR, "cache pooled event destruction failed");
            cache->free_events.pop_back();
        }
        ddo_vmm_host_ring_destroy(cache->ring);
        {
            std::lock_guard<std::mutex> lock(aggregate_mutex);
            aggregate_limits.erase(cache);
            refresh_ring_guard();
        }
        cuDevicePrimaryCtxRelease(cache->device);
        delete cache; *owner = nullptr;
        return DDO_VMM_SUCCESS;
    });
}

namespace {
ddo_vmm_status reap_batch(ddo_vmm_source_batch* batch, bool wait) {
    ApiTimer timer{batch->cache, &ddo_vmm_source_cache_stats::batch_reap_seconds};
    for (size_t i = 0; i < batch->pending.size();) {
        auto*& ticket = batch->pending[i];
        int ready = 1;
        if (!wait && ticket->submitted) {
            auto status = ddo_vmm_source_ticket_dma_ready(ticket, &ready);
            if (status != DDO_VMM_SUCCESS) return status;
        }
        if (!ready) { ++i; continue; }
        auto status = ddo_vmm_source_ticket_close(&ticket);
        if (status != DDO_VMM_SUCCESS) return status;
        batch->pending.erase(batch->pending.begin() + i);
    }
    return DDO_VMM_SUCCESS;
}
}

extern "C" ddo_vmm_status ddo_vmm_source_batch_create(ddo_vmm_source_cache* cache,
    const ddo_vmm_source_span* spans, size_t count, ddo_vmm_host_ring* ring,
    size_t ring_slots, size_t ring_slot_bytes, ddo_vmm_source_batch** output) {
    return boundary([&]() -> ddo_vmm_status {
        if (!output) return failure(DDO_VMM_INVALID_ARGUMENT, "batch output required");
        *output = nullptr;
        if (!cache || !spans || !count || count > 65536 || !ring || !ring_slots || !ring_slot_bytes)
            return failure(DDO_VMM_INVALID_ARGUMENT, "batch requires bounded immutable spans and fallback ring");
        auto status = context_check(cache);
        if (status != DDO_VMM_SUCCESS) return status;
        std::lock_guard<std::recursive_mutex> lock(cache->batch_mutex);
        auto batch = std::make_unique<ddo_vmm_source_batch>();
        batch->cache = cache; batch->ring = ring;
        batch->ring_slots = ring_slots; batch->ring_slot_bytes = ring_slot_bytes;
        batch->spans.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            const auto& input = spans[i];
            if (!input.source || !input.size || input.size > SIZE_MAX - input.dest_offset)
                return failure(DDO_VMM_INVALID_ARGUMENT, "invalid immutable batch span");
            ddo_vmm_source_file_info info{};
            info.struct_size = sizeof(info); info.version = 1;
            status = ddo_vmm_source_file_get_info(input.source, &info);
            if (status != DDO_VMM_SUCCESS || input.file_offset > info.size_bytes
                || input.size > info.size_bytes - input.file_offset)
                return failure(DDO_VMM_INVALID_ARGUMENT, "batch span exceeds immutable source");
            ddo_vmm_source_span span = input;
            span.source = nullptr;
            status = ddo_vmm_source_file_retain(input.source, &span.source);
            if (status != DDO_VMM_SUCCESS) return failure(status, ddo_vmm_source_last_error());
            batch->spans.push_back(span);
        }
        if (cuEventCreate(&batch->completion, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS)
            return failure(DDO_VMM_CUDA_ERROR, "batch completion event creation failed");
        cache->batches.push_back(batch.get());
        *output = batch.release();
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_cache_collect_completed(ddo_vmm_source_cache* cache) {
    return boundary([&]() -> ddo_vmm_status {
        if (!cache) return failure(DDO_VMM_INVALID_ARGUMENT, "live source cache required");
        auto status = context_check(cache);
        if (status != DDO_VMM_SUCCESS) return status;
        // Same lock ordering and reaper as submission; never steals external
        // ticket ownership, waits for pending work or changes storage ceilings.
        std::lock_guard<std::recursive_mutex> lock(cache->batch_mutex);
        for (auto* batch : cache->batches) {
            status = reap_batch(batch, false);
            if (status != DDO_VMM_SUCCESS) return status;
        }
        return DDO_VMM_SUCCESS;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_batch_submit(ddo_vmm_source_batch* batch,
    ddo_vmm_model_arena* arena, size_t dest_base, void* stream) {
    return boundary([&]() -> ddo_vmm_status {
        if (!batch || !arena) return failure(DDO_VMM_INVALID_ARGUMENT, "live batch and destination required");
        auto* cache = batch->cache;
        ApiTimer timer{cache, &ddo_vmm_source_cache_stats::batch_submit_seconds};
        std::lock_guard<std::recursive_mutex> lock(cache->batch_mutex);
        auto context_status = context_check(cache);
        if (context_status != DDO_VMM_SUCCESS) return context_status;
        if (batch->unsafe_completion) return failure(DDO_VMM_CUDA_ERROR, "batch completion is unproven");
        auto submit_work = [&]() -> ddo_vmm_status {
        ddo_vmm_detail::TimelineScope timing(1, reinterpret_cast<uintptr_t>(batch), 0, stream);
        // Recycle completed transfers across every unit, not only on the next
        // model pass. This bounds event retention while compute stays async.
        for (auto* previous : cache->batches) {
            auto status = reap_batch(previous, false);
            if (status != DDO_VMM_SUCCESS) return status;
        }
        size_t windows = 0;
        for (const auto& span : batch->spans) {
            if (span.dest_offset > SIZE_MAX - dest_base
                || span.size > ddo_vmm_model_arena_virtual_size(arena)
                || dest_base + span.dest_offset > ddo_vmm_model_arena_virtual_size(arena) - span.size)
                return failure(DDO_VMM_INVALID_ARGUMENT, "batch destination exceeds arena");
            const auto whole_windows = span.size / cache->slot_size;
            if (whole_windows > 1048574 || windows > 1048574 - whole_windows)
                return failure(DDO_VMM_INVALID_ARGUMENT, "batch window count too large");
            windows += whole_windows + 2;
        }
        if (windows > 1048576) return failure(DDO_VMM_INVALID_ARGUMENT, "batch window count too large");
        batch->pending.reserve(batch->pending.size() + windows);
        struct Work { const ddo_vmm_source_file* source; uint64_t offset; size_t size, dest; ddo_vmm_source_ticket* ticket; };
        std::vector<Work> work;
        work.reserve(64);
        auto flush = [&]() -> ddo_vmm_status {
            for (const auto& item : work) {
                if (item.ticket) {
                    auto status = ddo_vmm_source_ticket_copy_async(item.ticket, arena, item.dest, stream);
                    if (status == DDO_VMM_BYPASS) {
                        const size_t slot = batch->next_slot++ % batch->ring_slots;
                        status = ddo_vmm_host_ring_wait_slot(batch->ring, slot);
                        if (status == DDO_VMM_SUCCESS)
                            status = ddo_vmm_source_ticket_copy_via_ring_async(item.ticket, batch->ring, slot, arena, item.dest, stream);
                    }
                    if (status != DDO_VMM_SUCCESS) return status;
                } else {
#if defined(_WIN32) || defined(__linux__)
                    size_t copied = 0;
                    while (copied < item.size) {
                        const size_t bytes = std::min(batch->ring_slot_bytes, item.size - copied);
                        const size_t slot = batch->next_slot++ % batch->ring_slots;
                        auto status = ddo_vmm_host_ring_wait_slot(batch->ring, slot);
                        if (status == DDO_VMM_SUCCESS)
                            status = ddo_source_io::read_source_async(batch->ring, slot, item.source, item.offset + copied, bytes);
                        if (status == DDO_VMM_SUCCESS) status = ddo_vmm_host_ring_wait_file_read(batch->ring, slot);
                        if (status == DDO_VMM_SUCCESS)
                            status = ddo_vmm_host_ring_copy_to_arena_async(batch->ring, slot, 0, arena, item.dest + copied, bytes, stream);
                        if (status != DDO_VMM_SUCCESS) return status;
                        copied += bytes;
                    }
                    std::lock_guard<std::mutex> stats_lock(cache->mutex);
                    cache->stats.batch_bypass_bytes += item.size;
#else
                    return failure(DDO_VMM_UNSUPPORTED, "native source batch requires Windows");
#endif
                }
            }
            work.clear();
            return DDO_VMM_SUCCESS;
        };
        for (const auto& span : batch->spans) {
            size_t copied = 0;
            while (copied < span.size) {
                const uint64_t offset = span.file_offset + copied;
                const size_t size = std::min(span.size - copied, cache->slot_size - static_cast<size_t>(offset % cache->slot_size));
                ddo_vmm_source_ticket* ticket = nullptr;
                auto status = ddo_vmm_source_cache_request(cache, span.source, offset, size, &ticket);
                if (status != DDO_VMM_SUCCESS && status != DDO_VMM_BYPASS) return status;
                if (ticket) batch->pending.push_back(ticket);
                work.push_back({span.source, offset, size, dest_base + span.dest_offset + copied, ticket});
                copied += size;
                if (work.size() == 64) {
                    status = flush();
                    if (status != DDO_VMM_SUCCESS) return status;
                }
            }
        }
        auto status = flush();
        if (status == DDO_VMM_SUCCESS) {
            std::lock_guard<std::mutex> stats_lock(cache->mutex);
            ++cache->stats.batch_submissions;
        }
        return status;
        };
        ddo_vmm_status status;
        try {
            status = submit_work();
        } catch (...) {
            // An allocation/host exception may follow a successful DMA enqueue.
            // Fence partial work before boundary() translates the exception.
            batch->unsafe_completion = true;
            if (cuEventRecord(batch->completion, static_cast<CUstream>(stream)) == CUDA_SUCCESS) {
                batch->completion_recorded = true;
                batch->unsafe_completion = false;
            }
            throw;
        }
        // Include bypass DMA and partial submission failures in the batch's
        // one persistent completion event. Failed recording retains ownership.
        batch->unsafe_completion = true;
        if (cuEventRecord(batch->completion, static_cast<CUstream>(stream)) != CUDA_SUCCESS)
            return failure(DDO_VMM_CUDA_ERROR, "batch completion recording failed");
        batch->completion_recorded = true;
        batch->unsafe_completion = false;
        return status;
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_batch_wait_on_stream(ddo_vmm_source_batch* batch, void* stream) {
    return boundary([&]() -> ddo_vmm_status {
        if (!batch || !batch->completion_recorded || batch->unsafe_completion)
            return failure(DDO_VMM_INVALID_ARGUMENT, "batch has no safe completion event");
        auto status = context_check(batch->cache);
        if (status != DDO_VMM_SUCCESS) return status;
        std::lock_guard<std::recursive_mutex> lock(batch->cache->batch_mutex);
        ddo_vmm_detail::TimelineScope timing(2, reinterpret_cast<uintptr_t>(batch), 0, stream);
        return cuda_status(cuStreamWaitEvent(static_cast<CUstream>(stream), batch->completion, 0), "batch compute dependency failed");
    });
}

extern "C" ddo_vmm_status ddo_vmm_source_batch_close(ddo_vmm_source_batch** owner) {
    return boundary([&]() -> ddo_vmm_status {
        if (!owner) return failure(DDO_VMM_INVALID_ARGUMENT, "batch owner required");
        auto* batch = *owner;
        if (!batch) return DDO_VMM_SUCCESS;
        auto* cache = batch->cache;
        {
            std::lock_guard<std::recursive_mutex> lock(cache->batch_mutex);
            if (batch->unsafe_completion) return failure(DDO_VMM_CUDA_ERROR, "batch completion is unproven; ownership retained");
            if (batch->completion_recorded && cuEventSynchronize(batch->completion) != CUDA_SUCCESS)
                return failure(DDO_VMM_CUDA_ERROR, "batch still owns live DMA");
            auto status = reap_batch(batch, true);
            if (status != DDO_VMM_SUCCESS) return status;
            auto& batches = cache->batches;
            batches.erase(std::remove(batches.begin(), batches.end(), batch), batches.end());
        }
        delete batch; *owner = nullptr;
        return DDO_VMM_SUCCESS;
    });
}
