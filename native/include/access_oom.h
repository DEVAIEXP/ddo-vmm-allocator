#pragma once

namespace ddo_vmm_detail {
enum class AccessRollback { fatal, evictable_oom, retain };

template<class Status, class Unmap, class Release, class FreeAddress, class ReleaseContext>
AccessRollback rollback_failed_access(Status access_status, Status success, Status oom,
                                      bool barrier_failed, Unmap unmap, Release release,
                                      FreeAddress free_address, ReleaseContext release_context) {
    // The caller retains ownership of any resources that have not been released.
    if (barrier_failed) return AccessRollback::retain;
    if (unmap() != success || release() != success ||
        free_address() != success || release_context() != success) {
        return AccessRollback::retain;
    }
    return access_status == oom ? AccessRollback::evictable_oom : AccessRollback::fatal;
}
}
