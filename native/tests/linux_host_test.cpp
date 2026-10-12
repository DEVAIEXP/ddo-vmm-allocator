#include "host_platform.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void write(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path);
    output << text;
    require(bool(output), "fixture write failed");
}
struct Fixture {
    std::filesystem::path root;
    Fixture() {
        char path[] = "/tmp/ddo-linux-host-XXXXXX";
        auto created = mkdtemp(path);
        require(created != nullptr, "fixture create failed");
        root = created;
    }
    ~Fixture() { std::filesystem::remove_all(root); }
};
}
int main() {
    try {
        using namespace ddo_linux_memory;
        uint64_t ram = 0, commit = 0, available = 0;
        require(system_headroom("MemAvailable: 100 kB\nCommitLimit: 80 kB\nCommitted_AS: 30 kB\n", "2", ram, commit), "strict-mode meminfo parse");
        require(ram == 102400 && commit == 51200, "RAM/commit must stay distinct");
        require(system_headroom("MemAvailable: 100 kB\nCommitLimit: 10 kB\nCommitted_AS: 30 kB\n", "2", ram, commit) && commit == 0, "saturating strict-mode commit");
        const std::string oversubscribed = "MemAvailable: 100 kB\nCommitLimit: 10 kB\nCommitted_AS: 30 kB\n";
        for (const auto* mode : {"0", "1"}) {
            require(system_headroom(oversubscribed, mode, ram, commit)
                && ram == 102400 && commit == ram, "non-strict admission remains physically bounded");
            require(!ddo_host_platform::permits({ram, commit}, ram + 1, 0), "mode 0/1 never allows unlimited growth");
            require(!ddo_host_platform::permits({ram, commit}, ram, 1), "mode 0/1 retains headroom guard");
        }
        for (const auto* mode : {"", "unknown", "3", "-1", "18446744073709551616"})
            require(!system_headroom(oversubscribed, mode, ram, commit), "unknown/malformed mode refuses admission");
        require(!system_headroom("MemAvailable: 100 kB\n", "2", ram, commit), "missing commit must refuse");
        require(!system_headroom("MemAvailable: 100 kB\n", "0", ram, commit), "non-strict mode does not conceal missing metadata");
        require(!kib_field("MemAvailable: 18446744073709551615 kB", "MemAvailable", ram), "overflow rejected");
        require(remaining("max", "0", available) && available == UINT64_MAX, "unlimited cgroup");
        require(remaining("100", "110", available) && available == 0, "over-cap cgroup");
        require(!remaining("100", "bad", available), "unknown usage refused");
        Fixture fixture;
        const auto child = fixture.root / "child";
        const auto leaf = child / "leaf";
        std::filesystem::create_directories(leaf);
        write(fixture.root / "cgroup.controllers", "memory cpu");
        write(child / "cgroup.controllers", "memory cpu");
        write(child / "memory.max", "1000");
        write(child / "memory.current", "800");
        write(child / "memory.high", "max");
        write(leaf / "memory.max", "500");
        write(leaf / "memory.current", "50");
        write(leaf / "memory.high", "100");
        const auto mounts = "1 0 0:1 / " + fixture.root.string() + " rw - cgroup2 cgroup2 rw\n";
        require(cgroup_headroom("0::/\n", mounts, available) && available == UINT64_MAX,
            "root membership without memory.max preserves mount identity");
        require(cgroup_headroom("0::/child/leaf\n", mounts, available) && available == 50, "leaf high bound");
        write(leaf / "memory.high", "max");
        require(cgroup_headroom("0::/child/leaf\n", mounts, available) && available == 200, "ancestor capacity bound");
        const auto namespace_mount = "1 0 0:1 /child " + child.string() + " rw - cgroup2 cgroup2 rw\n";
        require(cgroup_headroom("0::/child\n", namespace_mount, available) && available == 200,
            "exact subtree mount-root membership retains memory limits");
        require(cgroup_headroom("0::/child/leaf\n", namespace_mount, available) && available == 200, "subtree mount root");
        write(child / "memory.current", "bad");
        require(!cgroup_headroom("0::/child/leaf\n", mounts, available), "malformed ancestor refused");
        require(!cgroup_headroom("0::/../../escape\n", mounts, available), "namespace escape refused");
        require(!cgroup_headroom("1:memory:/child\n", mounts, available), "unobserved v1 refused");
        require(!cgroup_headroom("0::/\n", "1 0 0:1 / /nonexistent/ddo-cgroup rw - cgroup2 cgroup2 rw\n", available), "missing hierarchy refused");
        ddo_host_platform::Headroom memory{};
        require(ddo_host_platform::headroom(memory), "real Linux host/cgroup observation");
        require(ddo_host_platform::permits({1024, 2048}, 512, 512), "headroom exact fit");
        require(!ddo_host_platform::permits({1024, 2048}, 512, 513), "headroom deficit");
        require(!ddo_host_platform::permits({1024, 2048}, UINT64_MAX, 1), "headroom overflow");
        constexpr size_t size = 1024 * 1024;
        auto* bytes = static_cast<unsigned char*>(ddo_host_platform::allocate(size));
        require(bytes != nullptr, "native anonymous storage");
        bytes[0] = 17; bytes[size - 1] = 43;
        require(bytes[0] == 17 && bytes[size - 1] == 43, "storage byte lifetime");
        require(ddo_host_platform::release(bytes, size), "native storage release");
        std::cout << "Linux host/storage CPU contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
