#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <tuple>

namespace ddo_vmm_detail {

// Metadata only; externally serialized. No allocation of source storage,
// file I/O, CUDA, blocking waits or worker creation. The native coordinator
// supplies existing capacity/queue bounds, admission and retained read owners.
struct SourceWindowKey {
    uint64_t volume = 0, file = 0, file_size = 0, modified = 0;
    uint64_t offset = 0, bytes = 0;

    bool valid() const {
        return bytes != 0 && offset <= file_size && bytes <= file_size - offset;
    }
    auto fields() const {
        return std::tie(volume, file, file_size, modified, offset, bytes);
    }
    bool operator<(const SourceWindowKey& other) const { return fields() < other.fields(); }
};

enum class SourceReadState { queued, reading, ready, failed };
enum class SourceReadAdmission { accepted, existing, deferred, closed };

struct SourceReadToken {
    uint64_t owner = 0, generation = 0;
};

class SourceReadScheduler {
public:
    struct Work {
        SourceReadToken token;
        SourceWindowKey key;
        // Must retain immutable file AND destination storage for all chunks.
        // This owner is never mutated or dropped by cancellation of a worker.
        std::shared_ptr<const void> read_owner;
    };
    struct Admission {
        SourceReadAdmission status;
        SourceReadToken token{};
    };
    struct Snapshot {
        size_t queued = 0, reading = 0, ready = 0, failed = 0;
        bool closing = false;
    };

    SourceReadScheduler(uint64_t owner, size_t queue_limit, size_t entry_limit)
        : owner_(owner), queue_limit_(queue_limit), entry_limit_(entry_limit) {
        if (!owner || !queue_limit || !entry_limit)
            throw std::invalid_argument("source scheduler requires owner and bounded capacities");
    }
    SourceReadScheduler(const SourceReadScheduler&) = delete;
    SourceReadScheduler& operator=(const SourceReadScheduler&) = delete;

    // A coordinator must prove host admission BEFORE providing read_owner.
    // Deferred requests keep no lease; demand can immediately use the existing
    // fallback. This scheduler never waits for capacity or increases a ceiling.
    Admission request(SourceWindowKey key, std::shared_ptr<const void> read_owner,
        bool demand) {
        if (!key.valid() || !read_owner)
            throw std::invalid_argument("source read needs a valid immutable window and owner");
        if (closing_) return {SourceReadAdmission::closed};
        auto found = entries_.find(key);
        if (found != entries_.end()) {
            found->second.demand = found->second.demand || demand;
            // Cancellation never interrupted active I/O. A new required
            // consumer may reclaim that same read before acknowledgement;
            // failure still publishes failed, never synthetic readiness.
            if (demand) found->second.cancelled = false;
            return {SourceReadAdmission::existing, found->second.work->token};
        }
        if (next_ == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("source scheduler generations exhausted");

        // Allocate and publish the new metadata before removing speculation.
        // Host allocation failure therefore cannot lose an existing request.
        auto work = std::make_shared<Work>(Work{{owner_, next_ + 1}, key, std::move(read_owner)});
        auto victim = entries_.end();
        if (queued_ == queue_limit_ || entries_.size() == entry_limit_) {
            if (demand) {
                for (auto it = entries_.begin(); it != entries_.end(); ++it) {
                    if (it->second.state == SourceReadState::queued && !it->second.demand
                        && (victim == entries_.end()
                            || it->second.work->token.generation > victim->second.work->token.generation))
                        victim = it;
                }
            }
            if (victim == entries_.end()) return {SourceReadAdmission::deferred};
        }
        entries_.emplace(key, Entry{work, SourceReadState::queued, demand, false});
        ++next_;
        if (victim != entries_.end()) { entries_.erase(victim); --queued_; }
        ++queued_;
        return {SourceReadAdmission::accepted, work->token};
    }

    // Caller chooses when a worker is available; this does not enqueue chunks.
    // Required demand always precedes speculation, independent of discovery
    // order, with FIFO generation ordering within either priority class.
    std::shared_ptr<const Work> take() {
        if (closing_) return {};
        auto best = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->second.state != SourceReadState::queued) continue;
            if (best == entries_.end()
                || (it->second.demand && !best->second.demand)
                || (it->second.demand == best->second.demand
                    && it->second.work->token.generation < best->second.work->token.generation))
                best = it;
        }
        if (best == entries_.end()) return {};
        best->second.state = SourceReadState::reading;
        --queued_;
        return best->second.work;
    }

    // Only acknowledge after ALL queued chunks are known complete. A partial
    // enqueue failure is not completion proof: cancel() retains reading work.
    // Worker ownership remains strong even after this ledger retires an entry.
    bool acknowledge(SourceReadToken token, bool succeeded) {
        auto entry = lookup(token);
        if (entry == entries_.end() || entry->second.state != SourceReadState::reading) return false;
        if (entry->second.cancelled || closing_) entries_.erase(entry);
        else entry->second.state = succeeded ? SourceReadState::ready : SourceReadState::failed;
        return true;
    }

    bool cancel(SourceReadToken token) {
        auto entry = lookup(token);
        if (entry == entries_.end()) return false;
        if (entry->second.state == SourceReadState::reading) entry->second.cancelled = true;
        else {
            if (entry->second.state == SourceReadState::queued) --queued_;
            entries_.erase(entry);
        }
        return true;
    }

    bool state(SourceReadToken token, SourceReadState& result) const {
        if (token.owner != owner_) return false;
        for (const auto& [key, entry] : entries_) {
            if (entry.work->token.generation == token.generation) {
                result = entry.state;
                return true;
            }
        }
        return false;
    }

    void begin_close() {
        closing_ = true;
        for (auto it = entries_.begin(); it != entries_.end();) {
            if (it->second.state == SourceReadState::reading) {
                it->second.cancelled = true;
                ++it;
            } else it = entries_.erase(it);
        }
        queued_ = 0;
    }
    bool drained() const { return closing_ && entries_.empty(); }
    Snapshot snapshot() const {
        Snapshot result{};
        result.closing = closing_;
        for (const auto& [key, entry] : entries_) {
            switch (entry.state) {
            case SourceReadState::queued: ++result.queued; break;
            case SourceReadState::reading: ++result.reading; break;
            case SourceReadState::ready: ++result.ready; break;
            case SourceReadState::failed: ++result.failed; break;
            }
        }
        return result;
    }

private:
    struct Entry {
        std::shared_ptr<const Work> work;
        SourceReadState state;
        bool demand, cancelled;
    };
    using Entries = std::map<SourceWindowKey, Entry>;
    Entries::iterator lookup(SourceReadToken token) {
        if (token.owner != owner_) return entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it)
            if (it->second.work->token.generation == token.generation) return it;
        return entries_.end();
    }
    uint64_t owner_, next_ = 0;
    size_t queue_limit_, entry_limit_, queued_ = 0;
    bool closing_ = false;
    Entries entries_;
};

}  // namespace ddo_vmm_detail
