#include "ddo_vmm_cache_selection.h"
#include <cstdio>
#include <limits>
#include <map>

int main() {
    struct Record { int device_ordinal; int id; };
    std::multimap<size_t, Record> cache;
    size_t failures = 0;
    auto require = [&](bool value) { failures += !value; };
    auto find = [&](size_t bytes, int device) {
        return ddo_vmm_detail::find_exact_cached_allocation(cache, bytes, device);
    };
    require(find(2, 0) == cache.end());
    cache.emplace(1, Record{0, 1});
    cache.emplace(2, Record{1, 2});
    cache.emplace(8, Record{0, 3});
    require(find(2, 0) == cache.end()); // Wrong device cannot borrow larger backing.
    require(find(4, 0) == cache.end()); // Neither smaller nor larger size qualifies.
    cache.emplace(2, Record{0, 4});
    const auto selected = find(2, 0);
    require(selected != cache.end() && selected->second.id == 4);
    require(find(2, 1) != cache.end() && find(2, 1)->second.id == 2);
    require(cache.size() == 4); // Selection is read-only, no eviction or release.
    if (selected != cache.end()) cache.erase(selected);
    require(find(2, 0) == cache.end() && cache.size() == 3);
    const auto maximum = std::numeric_limits<size_t>::max();
    cache.emplace(maximum, Record{0, 5});
    require(find(maximum, 0) != cache.end());
    require(find(maximum - 1, 0) == cache.end()); // No ratio/rounding overflow.
    std::printf("Exact cache selection CPU contracts: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures ? 1 : 0;
}
