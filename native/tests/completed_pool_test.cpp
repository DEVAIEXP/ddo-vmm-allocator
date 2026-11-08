#include "ddo_vmm_completed_backing.h"
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <thread>

namespace {
using namespace ddo_vmm_detail;
int checks = 0;
void require(bool value) {
    ++checks;
    if (!value) { std::cerr << "FAIL check " << checks << '\n'; std::exit(1); }
}
struct Fence : SpanFence { SpanCompletion result = SpanCompletion::not_ready; };
struct Owner {
    int device = 0;
    void* context = reinterpret_cast<void*>(uintptr_t{4096});
    bool retiring = false;
    SpanLedger ledger{4096, 0, 1};
};
struct Pool {
    std::mutex mutex;
    std::vector<std::shared_ptr<Owner>> owners;
    int queries = 0, releases = 0, retires = 0, observations = 0;
    bool fail_release = false, throw_release = false, nested = false;
    std::shared_ptr<Owner> add() {
        auto owner = std::make_shared<Owner>(); owners.push_back(owner); return owner;
    }
    ArenaPoolRelease run(size_t required, int device = 0, uintptr_t context = 4096) {
        return release_completed_pool(mutex, owners, device, context, required,
            [&](Owner&, const SpanFenceOwner& dependency) {
                ++queries; return std::static_pointer_cast<const Fence>(dependency)->result;
            }, [&](Owner&) {
                ++releases;
                if (nested) {
                    const auto inner = run(4096);
                    require(inner.status == CompletedRelease::unavailable && inner.bytes == 0);
                }
                if (throw_release) throw std::runtime_error("injected release failure");
                return !fail_release;
            }, [&](Owner&, CompletedRelease, size_t) { ++observations; },
            [&](Owner&) {
                // Another thread proves that registry/VA retirement runs with
                // the pool lock released, even if the caller owns an arena lock.
                auto unlocked = std::async(std::launch::async, [&] {
                    std::unique_lock lock(mutex, std::try_to_lock); return lock.owns_lock();
                });
                require(unlocked.get()); ++retires;
            });
    }
};
std::shared_ptr<Fence> pending(Owner& owner) {
    auto lease = owner.ledger.reserve(4096, 0);
    require(lease && owner.ledger.publish(*lease, true));
    auto fence = std::make_shared<Fence>();
    require(owner.ledger.retire(*lease, fence));
    return fence;
}
}
int main() {
    {
        Pool pool; for (int i = 0; i < 20; ++i) pool.add();
        const auto result = pool.run(20 * 4096);
        require(result.status == CompletedRelease::released);
        require(result.backings == 16 && result.bytes == 16 * 4096);
        require(pool.owners.size() == 4 && pool.releases == 16 && pool.retires == 16);
        require(pool.queries == 0 && pool.observations == 16);
    }
    {
        Pool pool; for (int i = 0; i < 4; ++i) pool.add();
        const auto result = pool.run(8192);
        require(result.backings == 2 && result.bytes == 8192 && pool.owners.size() == 2);
    }
    {
        Pool pool;
        auto foreign_device = pool.add(); foreign_device->device = 1;
        auto foreign_context = pool.add(); foreign_context->context = reinterpret_cast<void*>(uintptr_t{8192});
        auto retiring = pool.add(); retiring->retiring = true;
        auto live = pool.add(); auto lease = live->ledger.reserve(512, 0);
        require(lease && live->ledger.publish(*lease, true));
        pool.add();
        const auto result = pool.run(4096);
        require(result.backings == 1 && pool.owners.size() == 4 && pool.releases == 1);
        require(foreign_device->ledger.snapshot().mapped == 4096);
        require(foreign_context->ledger.snapshot().mapped == 4096);
        require(retiring->ledger.snapshot().mapped == 4096 && live->ledger.snapshot().leased == 512);
    }
    {
        Pool pool; auto busy = pool.add(); auto fence = pending(*busy); pool.add();
        const auto result = pool.run(4096);
        require(result.backings == 1 && pool.releases == 1 && pool.queries == 1);
        require(pool.owners.size() == 1 && busy->ledger.snapshot().pending == 4096);
    }
    {
        Pool pool; auto busy = pool.add(); auto fence = pending(*busy);
        fence->result = SpanCompletion::failed; pool.add();
        const auto result = pool.run(4096);
        require(result.status == CompletedRelease::failed && result.bytes == 0);
        require(pool.owners.size() == 2 && pool.releases == 0 && pool.retires == 0);
        require(busy->ledger.snapshot().quarantined == 4096);
    }
    for (bool throws : {false, true}) {
        Pool pool; auto owner = pool.add(); pool.add();
        pool.fail_release = true; pool.throw_release = throws;
        const auto result = pool.run(4096);
        require(result.status == CompletedRelease::failed && result.bytes == 0);
        require(pool.releases == 1 && pool.retires == 0 && pool.owners.size() == 2);
        require(owner->ledger.snapshot().quarantined == 4096 && !owner->retiring);
        require(!completed_pool_recovery_active);
    }
    {
        Pool pool; pool.add(); pool.nested = true;
        require(pool.run(4096).bytes == 4096 && pool.releases == 1);
    }
    {
        Pool pool; pool.add();
        std::unique_lock held(pool.mutex);
        auto contender = std::async(std::launch::async, [&] { return pool.run(4096); });
        require(contender.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
        const auto result = contender.get();
        require(result.status == CompletedRelease::unavailable && result.bytes == 0);
        require(pool.releases == 0 && pool.owners.size() == 1);
    }
    {
        Pool pool; pool.add();
        require(pool.run(0).status == CompletedRelease::unavailable);
        require(pool.run(4096, -1).status == CompletedRelease::unavailable);
        require(pool.run(4096, 0, 0).status == CompletedRelease::unavailable);
        require(pool.releases == 0);
    }
    std::cout << "PASS " << checks << " coordinator CPU assertions; no CUDA calls\n";
}
