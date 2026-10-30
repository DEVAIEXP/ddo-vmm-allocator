#include "ddo_vmm_mapping_trace.h"
#include <cstdlib>
int main() {
    uint64_t header[16]{};
    if (ddo_vmm_mapping_trace_start_v1(1, 128) != 2
        || ddo_vmm_mapping_trace_stop_v1() != 2
        || ddo_vmm_mapping_trace_snapshot_v1(header, 16, nullptr, 0) != 2) std::abort();
    for (auto value : header) if (value != 0) std::abort();
    return 0;
}
