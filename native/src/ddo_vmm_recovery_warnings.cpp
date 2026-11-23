#include "ddo_vmm.h"
#include "ddo_vmm_recovery_warnings.h"

extern "C" DDO_VMM_API int ddo_vmm_set_recovery_warns_v1(int enabled) {
    if (enabled != 0 && enabled != 1) return 1;
    ddo_vmm_detail::recovery_warnings_enabled.store(enabled != 0, std::memory_order_relaxed);
    return 0;
}
