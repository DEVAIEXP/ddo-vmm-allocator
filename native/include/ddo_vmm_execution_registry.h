#pragma once
#include "ddo_vmm_execution_reclaim.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

namespace execution_pressure {
using execution_reclaim::Result;
// Registry -> pool -> arena. Lookup never waits for an in-use registry/pool.
// Close excludes lookups until successful removal; failure retains the entry.
template<class Owner> class Registry {
    std::mutex mutex_;
    std::vector<Owner*> owners_;
public:
    void add(Owner* owner) {
        std::scoped_lock lock(mutex_);
        owners_.push_back(owner);
    }
    template<class Visit> Result visit(Visit visit) {
        std::unique_lock lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return Result::busy;
        for (auto* owner : owners_) {
            const auto result = visit(owner);
            switch (result) {
            case Result::busy: break;
            default: return result;
            }
        }
        return Result::busy;
    }
    template<class Close> Result close(Owner* owner, Close close) {
        std::scoped_lock lock(mutex_);
        const auto found = std::find(owners_.begin(), owners_.end(), owner);
        if (found == owners_.end()) return Result::failed;
        const auto result = close();
        switch (result) {
        case Result::success: owners_.erase(found); break;
        default: break;
        }
        return result;
    }
};
}
