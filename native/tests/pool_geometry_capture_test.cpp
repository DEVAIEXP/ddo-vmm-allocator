#include "ddo_vmm_pool_geometry.h"
#include "ddo_vmm_span_ledger.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using namespace ddo_vmm_detail;
constexpr size_t MiB = 1024 * 1024;
size_t checks = 0;
void require(bool ok) {
    ++checks;
    if (!ok) throw std::runtime_error("Geometry capture CPU contract failed");
}
struct Owner {
    SpanLedger ledger;
    void* context;
    uintptr_t address;
    int device;
    bool retiring = false;
    Owner(size_t bytes, uint64_t generation, void* ctx, int ordinal = 0)
        : ledger(bytes, ordinal, generation), context(ctx), address(generation * 4096), device(ordinal) {}
};
SpanLedger::Lease live(Owner& owner, size_t bytes) {
    const auto lease = owner.ledger.reserve_joined(bytes, owner.device);
    require(lease.has_value());
    require(owner.ledger.publish(*lease, true));
    return *lease;
}
int main() {
    int first_context = 0, other_context = 0;
    const auto context = reinterpret_cast<uintptr_t>(&first_context);
    std::vector<std::unique_ptr<Owner>> owners;
    PoolGeometryFailure<> output;
    require(!output.available);
    output.capture(owners, 688 * MiB, 0, context);
    require(output.available && output.count == 0 && output.total == 0 && !output.overflow);
    owners.push_back(std::make_unique<Owner>(1024 * MiB, 1, &first_context));
    auto lease = live(*owners[0], 400 * MiB);
    auto fence = std::make_shared<SpanFence>();
    require(owners[0]->ledger.retire(lease, fence));
    const auto references = fence.use_count();
    output.capture(owners, 688 * MiB, 0, context);
    require(fence.use_count() == references);
    require(output.count == 1 && output.rows[0][1] == 1);
    require(output.rows[0][6] == 1);
    require(output.rows[0][10] == 624 * MiB && output.rows[0][11] == 400 * MiB);
    require(output.rows[0][13] == 624 * MiB && output.rows[0][14] == 1024 * MiB);
    require(output.rows[0][15] == 0 && output.rows[0][16] == 1 && output.rows[0][17] == 0);
    owners[0]->ledger.observe(fence, SpanCompletion::failed);
    owners.push_back(std::make_unique<Owner>(1024 * MiB, 2, &other_context));
    owners.push_back(std::make_unique<Owner>(1024 * MiB, 3, &first_context, 1));
    owners.push_back(std::make_unique<Owner>(1024 * MiB, 4, &first_context));
    owners[3]->retiring = true;
    output.capture(owners, 688 * MiB, 0, context);
    require(output.count == 4 && !output.overflow);
    require(output.rows[0][12] == 400 * MiB && output.rows[0][16] == 0);
    require(output.rows[1][3] == 0 && output.rows[1][6] == 0);
    require(output.rows[2][4] == 0 && output.rows[2][6] == 0);
    require(output.rows[3][5] == 1 && output.rows[3][6] == 0);
    PoolGeometryFailure<1> bounded;
    bounded.capture(owners, 688 * MiB, 0, context);
    require(bounded.available && bounded.overflow && bounded.total == 4 && bounded.count == 1);
    require(!bounded.write(nullptr, 688 * MiB, 688 * MiB, 0, context, 0, 2, 2));
    FILE* file = nullptr;
#ifdef _MSC_VER
    require(tmpfile_s(&file) == 0 && file != nullptr);
#else
    file = std::tmpfile();
    require(file != nullptr);
#endif
    // Fill every cell with its longest uint64 spelling to verify the text bound.
    PoolGeometryFailure<> maximal;
    maximal.available = true;
    maximal.count = 64;
    maximal.total = 64;
    for (auto& row : maximal.rows) row.fill(UINT64_MAX);
    require(maximal.write(file, SIZE_MAX, SIZE_MAX, 0, UINTPTR_MAX, UINTPTR_MAX, 2, 2));
    require(std::fflush(file) == 0);
    const auto length = std::ftell(file);
    require(length > 0 && length < 32 * 1024);
    require(std::fclose(file) == 0);
    // Emit a parseable fixture using the same serializer as the failure path.
    require(output.write(stdout, 720896000, 721420288, 0, context, 0, 2, 2));
    std::cout << "{\"status\":\"pass\",\"checks\":" << checks
              << ",\"cuda\":false,\"callbacks\":0,\"payload_bytes\":0}\n";
}
