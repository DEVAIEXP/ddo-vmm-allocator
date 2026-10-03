#include "ddo_vmm_allocation_footprint.h"
#include <cstdio>
#include <limits>

int main() {
    using ddo_vmm_detail::AllocationFootprint;
    size_t failures = 0;
    auto require = [&](bool value) { failures += !value; };
    const AllocationFootprint exact{4096, 4096, 4096};
    require(exact.valid() && exact.granularity_padding() == 0 && exact.reuse_excess() == 0);
    const AllocationFootprint rounded{512, 2097152, 2097152};
    require(rounded.valid() && rounded.granularity_padding() == 2096640 && rounded.reuse_excess() == 0);
    const AllocationFootprint reused{512, 2097152, 8388608};
    require(reused.valid() && reused.granularity_padding() == 2096640 && reused.reuse_excess() == 6291456);
    require(reused.requested + reused.granularity_padding() + reused.reuse_excess() == reused.mapped);
    const AllocationFootprint zero{0, 2097152, 2097152};
    require(zero.valid() && zero.granularity_padding() == zero.mapped);
    require(!(AllocationFootprint{2, 1, 3}).valid());
    require(!(AllocationFootprint{1, 3, 2}).valid());
    const auto maximum = std::numeric_limits<size_t>::max();
    const AllocationFootprint large{maximum - 1, maximum, maximum};
    require(large.valid() && large.granularity_padding() == 1 && large.reuse_excess() == 0);
    std::printf("Allocation footprint CPU contracts: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures ? 1 : 0;
}
