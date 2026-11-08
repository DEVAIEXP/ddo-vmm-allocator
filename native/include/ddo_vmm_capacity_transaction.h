#pragma once
#include <cstddef>
#include <cstdint>

namespace ddo_vmm_detail {
// Internal (not exported) model-arena pressure used by the segmented capacity
// transaction. Model-arena pages only: these never fall back to whole-region
// pressure eviction. Lock order: pressure mutex, then arena mutation try-lock.
// The caller must not hold the Torch pool lock. No synchronization occurs.

struct ArenaPressureCount {
    size_t evictable_pages = 0; // Resident, unpinned pages of unlocked arenas.
    size_t busy_arenas = 0;     // Arenas skipped because their lock was held.
};
// Observation for the W4 bound, taken once at Tier 2 entry.
ArenaPressureCount pressure_count_arena_pages(int device) noexcept;

enum class ArenaEviction { evicted, none, failed };
struct ArenaEvictionResult {
    ArenaEviction outcome = ArenaEviction::none;
    size_t bytes = 0;       // Released page size, only when evicted.
    size_t busy_arenas = 0; // Arenas skipped because their lock was held.
};
// Release one eligible page with the existing arena selection policy. A
// Driver failure in page release is reported as failed and must stop Tier 2.
ArenaEvictionResult pressure_evict_arena_page(int device) noexcept;

// Observation only (experimental DDO_VMM_TIMING_COUNTERS): resident device
// bytes of registered model arenas, split into execution-slot arenas and the
// others, plus pressure regions. Never waits: the registry and each arena are
// try-locked; anything skipped makes the matching total unavailable.
struct ArenaComposition {
    bool registry_observed = false;
    size_t busy_arenas = 0;
    uint64_t model_arenas = 0, model_bytes = 0;  // Non-slot arenas.
    uint64_t slot_arenas = 0, slot_bytes = 0;    // Execution-slot arenas.
    uint64_t regions = 0, region_bytes = 0;
};
ArenaComposition observe_arena_composition(int device) noexcept;
}
