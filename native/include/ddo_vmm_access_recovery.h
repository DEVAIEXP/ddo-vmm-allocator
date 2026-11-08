#pragma once

namespace ddo_vmm_detail {
// Retry the existing mapping only after a successful context barrier.
template<class Status, class SetAccess, class Synchronize>
Status recover_access_once(Status success, Status not_ready,
                           SetAccess set_access, Synchronize synchronize) {
    const auto first = set_access();
    if (first != not_ready) return first;
    const auto barrier = synchronize();
    if (barrier != success) return barrier;
    return set_access();
}
}
