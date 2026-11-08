#pragma once

namespace ddo_vmm_detail {
enum class ArenaAccessDisposition { clean_oom, retained };

// A failed transition never publishes a page. Track each proven cleanup step
// independently so callers retain the exact remaining allocation/context owner.
struct ArenaAccessRollback {
    ArenaAccessDisposition disposition = ArenaAccessDisposition::retained;
    bool mapped = true;
    bool allocation_live = true;
    bool context_released = false;
};

template<class Status, class Unmap, class Release, class ReleaseContext>
ArenaAccessRollback rollback_arena_access(
    Status access_status, Status success, Status oom, bool barrier_failed,
    Unmap unmap, Release release, ReleaseContext release_context) {
    ArenaAccessRollback result;
    // Only an explicit allocation refusal permits rollback. A context/driver
    // error or failed completion barrier must retain all resources untouched.
    if (barrier_failed || access_status != oom) return result;
    if (unmap() != success) return result;
    result.mapped = false;
    if (release() != success) return result;
    result.allocation_live = false;
    if (release_context() != success) return result;
    result.context_released = true;
    result.disposition = ArenaAccessDisposition::clean_oom;
    return result;
}
}
