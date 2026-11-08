#include "segmented_driver_storage.h"
#include <iostream>
#include <windows.h>
std::mutex mutex;
int checks = 0, driver_calls = 0, admit_calls = 0, admission_result = 0;
size_t admission_bytes = 0;
bool lock_violation = false;
std::string fault;
void check(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
bool unlocked() noexcept {
    if (!mutex.try_lock()) { lock_violation = true; return false; }
    mutex.unlock(); return true;
}
int admission(int device, size_t bytes) noexcept {
    ++admit_calls; admission_bytes = bytes;
    if (!unlocked() || device != 0) return 999;
    return admission_result;
}
struct Driver {
    using Handle = unsigned;
    // Same shape as CudaSegmentDriver::AccessSequence; this double never reports
    // an access status sequence (the storage contract here does not exercise it).
    struct AccessSequence {
        bool first_observed = false, barrier_observed = false, continuation_observed = false;
        int first = 0, barrier = 0, continuation = 0;
    } access_sequence;
    const AccessSequence& last_access() const noexcept { return access_sequence; }
    Driver(uintptr_t, int) noexcept {}
    bool identity(int device) noexcept { return device == 0; }
    int step(const char* operation) noexcept {
        ++driver_calls;
        if (!unlocked()) return 999;
        return fault == operation ? 600 : 0;
    }
    int reserve(size_t, size_t, uintptr_t& address) noexcept {
        if (const int error = step("reserve")) return error;
        address = 0x100000; return 0;
    }
    int free_va(uintptr_t, size_t) noexcept { return step("free_va"); }
    int create(size_t, Handle& handle) noexcept {
        if (const int error = step("create")) return error;
        handle = 1; return 0;
    }
    int map(uintptr_t, size_t, Handle) noexcept { return step("map"); }
    int access(uintptr_t, size_t) noexcept { return step("access"); }
    int unmap(uintptr_t, size_t) noexcept { return step("unmap"); }
    int release(Handle) noexcept { return step("release"); }
};
using Storage = ddo_vmm_detail::SegmentedDriverStorage<Driver>;
auto fixture() {
    fault.clear(); admission_result = 0; admit_calls = driver_calls = 0; lock_violation = false;
    auto value = std::make_shared<Storage>(mutex, uintptr_t{0x200}, 0, 4096, 1024, 1, &admission);
    check(value->open() && value->va_charge() == 4096 && !value->physical_charge(), "VA-only open");
    return value;
}
int main() {
    try {
        { auto s = fixture(); auto& ledger = s->ledger(); std::unique_lock lock(mutex);
          auto lease = s->ledger().reserve(1024, 0);
          check(s->materialize(*lease, s->ledger(), lock) && lock.owns_lock(), "materialize restores lock");
          check(admit_calls == 1 && admission_bytes == 1024 && s->physical_charge() == 1024, "missing bytes admitted once");
          check(s->ledger().cancel_unsubmitted(*lease), "unsubmitted cancellation");
          lease = s->ledger().reserve(1024, 0);
          check(s->materialize(*lease, s->ledger(), lock) && admit_calls == 1, "resident reuse needs no admission");
          check(s->ledger().cancel_unsubmitted(*lease), "reuse cancellation");
          check(s->discard(0, s->ledger(), lock) && lock.owns_lock() && !s->physical_charge(), "release credits after success");
          check(s->ledger().begin_backing_release() && s->close_va(lock), "checked VA close");
          check(lock.owns_lock() && ledger.commit_backing_release() && !s->va_charge()
              && !s->error() && !lock_violation, "one ledger commit and clean accounting"); }
        { auto s = fixture(); auto& ledger = s->ledger(); admission_result = 3; std::unique_lock lock(mutex);
          auto lease = s->ledger().reserve(1024, 0); const int before = driver_calls;
          check(!s->materialize(*lease, s->ledger(), lock) && lock.owns_lock(), "admission refusal restores lock");
          check(s->admission_error() == 3 && !s->error() && !s->physical_charge()
              && driver_calls == before, "admission error distinct from Driver error");
          check(s->ledger().cancel_unsubmitted(*lease) && s->ledger().begin_backing_release()
              && s->close_va(lock) && ledger.commit_backing_release(), "refused lease can close safely"); }
        // Retain failed fake owners; CPU teardown is not a native cleanup proof.
        std::vector<std::shared_ptr<Storage>> retained;
        for (const std::string operation : {"create", "map", "access", "unmap", "release", "free_va"}) {
            auto s = fixture(); std::unique_lock lock(mutex);
            if (operation == "create" || operation == "map" || operation == "access") {
                fault = operation; auto lease = s->ledger().reserve(1024, 0);
                check(!s->materialize(*lease, s->ledger(), lock) && s->error() == 600
                    && s->physical_charge() == 1024, "creation failure retains original error and conservative charge");
            } else {
                auto lease = s->ledger().reserve(1024, 0);
                check(s->materialize(*lease, s->ledger(), lock) && s->ledger().cancel_unsubmitted(*lease), "release failure setup");
                fault = operation;
                check(s->ledger().begin_backing_release() && !s->close_va(lock) && s->error() == 600,
                    "failed close returns actual Driver error");
                check(s->va_charge() == 4096, "failed close retains VA charge");
            }
            check(lock.owns_lock() && !lock_violation, "failure lock and mutation ordering");
            retained.push_back(s);
        }
        check(!GetModuleHandleW(L"nvcuda.dll") && !GetModuleHandleW(L"cudart64_13.dll"), "no CUDA modules loaded");
        std::cout << "PASS native storage wrapper CPU: " << checks << " checks; no GPU\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
