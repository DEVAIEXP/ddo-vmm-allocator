#pragma once
#if defined(__linux__)
#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/resource.h>

namespace ddo_linux_memory {
inline bool number(const std::string& value, uint64_t& output) {
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), output);
    return !value.empty() && parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}
inline bool read_token(const std::filesystem::path& path, std::string& value) {
    std::ifstream input(path);
    return bool(input >> value);
}
inline bool kib_field(const std::string& text, const std::string& field, uint64_t& output) {
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string key, value, unit;
        if (!(row >> key) || key != field + ":") continue;
        if (!(row >> value >> unit) || unit != "kB" || !number(value, output)
            || output > UINT64_MAX / 1024) return false;
        output *= 1024;
        return true;
    }
    return false;
}
inline bool read_text(const std::filesystem::path& path, std::string& text) {
    std::ifstream input(path);
    if (!input) return false;
    text.assign(std::istreambuf_iterator<char>(input), {});
    return !input.bad();
}
inline bool system_headroom(const std::string& meminfo, const std::string& overcommit,
                            uint64_t& ram, uint64_t& commit) {
    uint64_t limit = 0, used = 0, mode = 0;
    if (!number(overcommit, mode) || mode > 2) return false;
    if (!kib_field(meminfo, "MemAvailable", ram)
        || !kib_field(meminfo, "CommitLimit", limit)
        || !kib_field(meminfo, "Committed_AS", used)) return false;
    // CommitLimit is enforced only in mode 2. Modes 0/1 still use a finite
    // physical-memory admission bound, not unlimited capacity or swap-backed
    // pinning. Both bounds are further restricted by cgroups/process limits.
    // Neither is a reservation; mmap/CUDA registration may still refuse bytes.
    commit = mode == 2 ? (used >= limit ? 0 : limit - used) : ram;
    return true;
}
inline bool remaining(const std::string& limit, const std::string& usage, uint64_t& result) {
    if (limit == "max") { result = UINT64_MAX; return true; }
    uint64_t maximum = 0, current = 0;
    if (!number(limit, maximum) || !number(usage, current)) return false;
    result = current >= maximum ? 0 : maximum - current;
    return true;
}
inline std::string unescape_mount(std::string text) {
    for (const auto& item : {std::pair{"\\040", " "}, {"\\011", "\t"},
                             {"\\012", "\n"}, {"\\134", "\\"}}) {
        size_t position = 0;
        while ((position = text.find(item.first, position)) != std::string::npos) {
            text.replace(position, 4, item.second);
            ++position;
        }
    }
    return text;
}
// v2 only for the initial port. Unknown/v1 memory controllers fail closed,
// rather than using host-wide capacity inside an unobserved container.
inline bool cgroup_headroom(const std::string& membership, const std::string& mounts,
                            uint64_t& bound) {
    std::string group;
    std::istringstream groups(membership);
    std::string line;
    while (std::getline(groups, line)) {
        if (line.rfind("0::", 0) == 0) group = line.substr(3);
        else if (line.find("memory") != std::string::npos) return false;
    }
    if (group.empty() || group[0] != '/') return false;
    std::istringstream table(mounts);
    while (std::getline(table, line)) {
        const auto separator = line.find(" - cgroup2 ");
        if (separator == std::string::npos) continue;
        std::istringstream row(line.substr(0, separator));
        std::string id, parent, device, root, mount;
        if (!(row >> id >> parent >> device >> root >> mount)) return false;
        root = unescape_mount(root); mount = unescape_mount(mount);
        if (root != "/" && group != root && group.rfind(root + "/", 0) != 0) continue;
        const auto relative = root == "/" ? group.substr(1) : group.substr(root.size());
        const auto base = std::filesystem::path(mount).lexically_normal();
        std::string root_controllers;
        if (!read_text(base / "cgroup.controllers", root_controllers)) return false;
        const auto directory = (base / std::filesystem::path(relative).relative_path()).lexically_normal();
        auto path = directory;
        bound = UINT64_MAX;
        for (size_t depth = 0; depth < 256; ++depth) {
            // Reject namespace escapes, missing hierarchy and unreadable limits.
            const auto rel = path.lexically_relative(base);
            if (rel.empty() || *rel.begin() == "..") return false;
            std::string maximum, current;
            if (read_token(path / "memory.max", maximum)) {
                if (!read_token(path / "memory.current", current)) return false;
                uint64_t available = 0;
                if (!remaining(maximum, current, available)) return false;
                bound = std::min(bound, available);
                // Respect memory.high as a conservative growth threshold too.
                std::string high;
                if (!read_token(path / "memory.high", high) || !remaining(high, current, available)) return false;
                bound = std::min(bound, available);
            } else if (path != base) {
                std::error_code error;
                if (std::filesystem::exists(path / "memory.max", error) || error) return false;
                std::string controllers;
                if (!read_text(path / "cgroup.controllers", controllers)
                    || controllers.find("memory") != std::string::npos) return false;
            }
            if (path == base) return true;  // v2 root has no memory.max.
            path = path.parent_path();
        }
        return false;
    }
    return false;
}
inline bool observe(uint64_t& ram, uint64_t& commit) noexcept {
    try {
        std::string meminfo, overcommit, groups, mounts, status;
        if (!read_token("/proc/sys/vm/overcommit_memory", overcommit)
            || !read_text("/proc/meminfo", meminfo)
            || !system_headroom(meminfo, overcommit, ram, commit)
            || !read_text("/proc/self/cgroup", groups) || !read_text("/proc/self/mountinfo", mounts)) return false;
        uint64_t cgroup = 0;
        if (!cgroup_headroom(groups, mounts, cgroup)) return false;
        ram = std::min(ram, cgroup); commit = std::min(commit, cgroup);
        // Anonymous mmap can also be limited by the process address/data limits.
        if (!read_text("/proc/self/status", status)) return false;
        for (const auto& item : {std::pair{RLIMIT_AS, "VmSize"}, {RLIMIT_DATA, "VmData"}}) {
            struct rlimit limit{};
            if (getrlimit(item.first, &limit) != 0) return false;
            if (limit.rlim_cur == RLIM_INFINITY) continue;
            uint64_t used = 0;
            if (!kib_field(status, item.second, used)) return false;
            const uint64_t available = used >= limit.rlim_cur ? 0 : limit.rlim_cur - used;
            ram = std::min(ram, available); commit = std::min(commit, available);
        }
        return true;
    } catch (...) { return false; }
}
}
#endif
