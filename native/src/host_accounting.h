#pragma once
#include <cstdint>
#include <limits>
#include <mutex>

// Process-local to this DLL. Cache admission and eager ring creation serialize
// registration through this mutex; workers never acquire it. External allocators
// are deliberately excluded and remain visible through system headroom checks.
namespace ddo_host_accounting {
inline std::mutex mutex;
inline uint64_t staging_registered_bytes = 0;
inline uint64_t cache_registered_bytes = 0;
inline uint64_t cache_committed_bytes = 0;
inline uint64_t active_cache_limit_bytes = std::numeric_limits<uint64_t>::max();
inline uint64_t active_cache_headroom_bytes = 0;
}
