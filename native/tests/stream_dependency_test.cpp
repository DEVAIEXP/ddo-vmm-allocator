#include "ddo_vmm_stream_dependency.h"
#include "ddo_vmm_timeline.h"

#include <cstdio>

int main() {
    int token = 1;
    int* event = &token;
    int waits = 0;
    int synchronizations = 0;
    int destroys = 0;
    auto wait = [&](int stream, int* actual, unsigned int flags) {
        ++waits;
        return stream == 7 && actual == &token && flags == 0 ? 0 : 1;
    };
    auto sync = [&](int* actual) { ++synchronizations; return actual == &token ? 0 : 1; };
    auto destroy = [&](int* actual) { ++destroys; return actual == &token ? 0 : 1; };
    auto require = [](bool condition) { return condition ? 0 : 1; };
    int failures = 0;
    const auto token_a = ddo_vmm_detail::timeline_token(1, 0);
    const auto token_b = ddo_vmm_detail::timeline_token(2, 0);
    failures += require(token_a != token_b);
    failures += require(ddo_vmm_detail::timeline_index(token_a, 1, 1) == 0);
    failures += require(ddo_vmm_detail::timeline_index(token_a, 2, 1) == SIZE_MAX);
    failures += require(ddo_vmm_detail::timeline_index(0, 1, 1) == SIZE_MAX);
    failures += require(ddo_vmm_detail::timeline_index(token_a, 1, 0) == SIZE_MAX);
    size_t samples = 0;
    uint64_t dropped = 0;
    failures += require(ddo_vmm_detail::admit_timeline_sample(2, samples, dropped) == 0);
    failures += require(ddo_vmm_detail::admit_timeline_sample(2, samples, dropped) == 1);
    failures += require(ddo_vmm_detail::admit_timeline_sample(2, samples, dropped) == SIZE_MAX);
    failures += require(samples == 2 && dropped == 1);
    failures += require(ddo_vmm_detail::enqueue_reuse_dependency(event, 7, wait));
    failures += require(waits == 1 && synchronizations == 0 && destroys == 0 && event == &token);
    failures += require(!ddo_vmm_detail::enqueue_reuse_dependency(event, 8, wait));
    failures += require(event == &token && synchronizations == 0 && destroys == 0);
    // An unpublished allocation must still drain its retained original fence.
    failures += require(!ddo_vmm_detail::drain_completion(event, [](int*) { return 1; }, destroy));
    failures += require(event == &token && destroys == 0);
    failures += require(ddo_vmm_detail::drain_completion(event, sync, destroy));
    failures += require(event == nullptr && synchronizations == 1 && destroys == 1);
    failures += require(ddo_vmm_detail::enqueue_reuse_dependency(event, 7, wait));
    failures += require(waits == 2);
    event = &token;
    const ddo_vmm_detail::StreamIdentity known{0, 0, true};
    // ID zero can be valid; known/unknown must not be inferred from its value.
    failures += require(ddo_vmm_detail::enqueue_reuse_dependency(event, 8, wait, known, known));
    failures += require(waits == 2 && event == &token && destroys == 1);
    failures += require(!ddo_vmm_detail::enqueue_reuse_dependency(event, 8, wait, known, {1, 0, true}));
    failures += require(!ddo_vmm_detail::enqueue_reuse_dependency(event, 8, wait, known, {0, 1, true}));
    failures += require(!ddo_vmm_detail::enqueue_reuse_dependency(event, 8, wait, known, {}));
    failures += require(!ddo_vmm_detail::enqueue_reuse_dependency(event, 8, wait, {}, known));
    failures += require(!ddo_vmm_detail::enqueue_reuse_dependency(event, 8, wait, {}, {}));
    failures += require(waits == 7 && event == &token && synchronizations == 1 && destroys == 1);
    // Skipping a redundant wait must not lose the fence on publication failure.
    failures += require(!ddo_vmm_detail::drain_completion(event, [](int*) { return 1; }, destroy));
    failures += require(event == &token && destroys == 1);
    failures += require(ddo_vmm_detail::drain_completion(event, sync, destroy));
    failures += require(event == nullptr && synchronizations == 2 && destroys == 2);
    failures += require(ddo_vmm_detail::drain_completion(event, sync, destroy));
    failures += require(synchronizations == 2 && destroys == 2);
    std::printf("Stream dependency CPU contracts: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
