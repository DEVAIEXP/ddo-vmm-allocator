#pragma once

namespace ddo_vmm_detail {

struct StreamIdentity {
    unsigned long long id = 0;
    int device = -1;
    bool known = false;
};

inline bool same_stream(StreamIdentity completion, StreamIdentity consumer) {
    return completion.known && consumer.known && completion.device >= 0
        && completion.device == consumer.device && completion.id == consumer.id;
}

// Keep the last-use fence alive after enqueueing. Allocation publication can
// still fail, in which case destruction must drain that original fence.
template <typename Event, typename Stream, typename Wait>
bool enqueue_reuse_dependency(Event event, Stream stream, Wait wait,
    StreamIdentity completion = {}, StreamIdentity consumer = {}) {
    // Stream order already protects reuse on the proven same stream. Never
    // infer identity from a handle: default/PTDS handles can alias.
    return event == nullptr || same_stream(completion, consumer)
        || wait(stream, event, 0) == 0;
}

// A failed completion proof must never permit backing to be unmapped.
template <typename Event, typename Synchronize, typename Destroy>
bool drain_completion(Event& event, Synchronize synchronize, Destroy destroy) {
    if (event == nullptr) {
        return true;
    }
    if (synchronize(event) != 0) {
        return false;
    }
    destroy(event);
    event = nullptr;
    return true;
}

}  // namespace ddo_vmm_detail
