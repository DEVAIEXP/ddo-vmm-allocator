#pragma once
#include "ddo_vmm.h"
#include <array>

namespace ddo_vmm_detail {
// Internal read-only pool identity. UINT64_MAX means unavailable, not zero.
enum OwnerField : size_t { owner_schema, owner_found, owner_query, owner_query_bytes,
    owner_backing, owner_generation, owner_lease, owner_requested, owner_offset,
    owner_lease_bytes, owner_device, owner_context, owner_retiring,
    owner_completion, owner_role, owner_version, owner_fields };
using OwnerObservation = std::array<uint64_t, owner_fields>;
struct OwnerMetadata {
    uint64_t backing, generation, lease, requested, offset, lease_bytes;
    uint64_t device, context, retiring;
};
// Lookup is injected for CPU contracts. It must be read-only and externally locked.
// Only the canonical allocation start is looked up. Interior views must supply
// their canonical storage address; absence never means there is no GPU owner.
template<class Lookup>
int observe_owner(uint64_t address, uint64_t bytes, uint64_t* output,
                  size_t count, Lookup&& lookup) {
    if (!address || !bytes || !output || count != owner_fields
        || bytes > UINT64_MAX-address) return 1;
    OwnerObservation row;
    row.fill(UINT64_MAX);
    row[owner_schema] = 1; row[owner_found] = 0;
    row[owner_query] = address; row[owner_query_bytes] = bytes;
    OwnerMetadata metadata{};
    if (lookup(address, metadata)) {
        if (!metadata.backing || !metadata.generation || !metadata.lease
            || !metadata.requested || metadata.requested > metadata.lease_bytes
            || metadata.offset > UINT64_MAX-metadata.backing
            || metadata.backing+metadata.offset != address
            || bytes > metadata.requested || metadata.device == UINT64_MAX
            || metadata.retiring > 1) return 1;
        row[owner_found] = 1; row[owner_backing] = metadata.backing;
        row[owner_generation] = metadata.generation; row[owner_lease] = metadata.lease;
        row[owner_requested] = metadata.requested; row[owner_offset] = metadata.offset;
        row[owner_lease_bytes] = metadata.lease_bytes; row[owner_device] = metadata.device;
        row[owner_context] = metadata.context; row[owner_retiring] = metadata.retiring;
    }
    for (size_t i=0; i<count; ++i) output[i] = row[i];
    return 0;
}
}

// Available only with mapping observation and the native span pool enabled.
// Return 0: valid snapshot (found can be 0); 1: invalid/error; 2: unavailable.
// This query does not establish version, role, GPU completion or VRAM placement.
extern "C" DDO_VMM_API int ddo_vmm_owner_observe_v1(
    uint64_t address, uint64_t bytes, uint64_t* output, size_t count);
