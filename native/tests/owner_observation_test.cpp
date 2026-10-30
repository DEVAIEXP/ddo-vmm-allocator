#include "ddo_vmm_owner_observation.h"
#include <cstdlib>
using namespace ddo_vmm_detail;
void require(bool ok) { if (!ok) std::abort(); }
int main() {
    OwnerObservation row;
    row.fill(99);
    unsigned calls=0;
    uint64_t generation=1;
    auto lookup = [&](uint64_t query, OwnerMetadata& m) {
        ++calls;
        if (query != 1100) return false;
        m = {1000,generation,7,80,100,128,0,55,0};
        return true;
    };
    require(observe_owner(1100,80,row.data(),row.size(),lookup)==0);
    require(calls==1 && row[owner_found]==1 && row[owner_backing]==1000);
    require(row[owner_generation]==1 && row[owner_lease]==7 && row[owner_device]==0);
    require(row[owner_role]==UINT64_MAX && row[owner_version]==UINT64_MAX
        && row[owner_completion]==UINT64_MAX);
    generation=2;
    require(observe_owner(1100,80,row.data(),row.size(),lookup)==0 && row[owner_generation]==2);
    require(observe_owner(1101,79,row.data(),row.size(),lookup)==0 && !row[owner_found]);
    require(row[owner_backing]==UINT64_MAX && row[owner_generation]==UINT64_MAX);
    const auto unchanged=row;
    require(observe_owner(1100,81,row.data(),row.size(),lookup)==1 && row==unchanged);
    const auto before=calls;
    require(observe_owner(UINT64_MAX,2,row.data(),row.size(),lookup)==1);
    require(observe_owner(1100,80,row.data(),row.size()-1,lookup)==1);
    require(observe_owner(0,80,row.data(),row.size(),lookup)==1 && calls==before);
    auto malformed=[](uint64_t, OwnerMetadata& m) { m={1000,0,7,80,100,128,0,55,0}; return true; };
    require(observe_owner(1100,80,row.data(),row.size(),malformed)==1 && row==unchanged);
}
