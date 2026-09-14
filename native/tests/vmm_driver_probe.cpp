#include "ddo_vmm.h"

#include <cuda.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

size_t parse_mebibytes(int argc, char** argv) {
    if (argc == 1) {
        return 64;
    }
    if (argc != 2) {
        throw std::runtime_error("usage: ddo_vmm_driver_probe [mebibytes]");
    }
    const long long value = std::stoll(argv[1]);
    if (value <= 0) {
        throw std::runtime_error("mebibytes must be positive");
    }
    return static_cast<size_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const size_t mebibytes = parse_mebibytes(argc, argv);
        const size_t bytes = mebibytes * 1024ULL * 1024ULL;

        ddo_vmm_region* region = nullptr;
        const ddo_vmm_status status = ddo_vmm_region_create(0, bytes, &region);
        if (status != DDO_VMM_SUCCESS) {
            std::cerr << "VMM region creation failed: " << ddo_vmm_last_error() << '\n';
            return EXIT_FAILURE;
        }

        const CUresult memset_result = cuMemsetD8(
            static_cast<CUdeviceptr>(ddo_vmm_region_address(region)), 0xA5,
            ddo_vmm_region_mapped_size(region)
        );
        if (memset_result != CUDA_SUCCESS || cuCtxSynchronize() != CUDA_SUCCESS) {
            std::cerr << "VMM region CUDA write failed\n";
            ddo_vmm_region_destroy(region);
            return EXIT_FAILURE;
        }

        const uintptr_t original_address = ddo_vmm_region_address(region);
        if (ddo_vmm_region_evict(region) != DDO_VMM_SUCCESS || ddo_vmm_region_is_materialized(region) != 0) {
            std::cerr << "VMM eviction failed: " << ddo_vmm_last_error() << '\n';
            ddo_vmm_region_destroy(region);
            return EXIT_FAILURE;
        }
        if (
            ddo_vmm_region_materialize(region) != DDO_VMM_SUCCESS
            || ddo_vmm_region_is_materialized(region) != 1
            || ddo_vmm_region_address(region) != original_address
        ) {
            std::cerr << "VMM rematerialization failed: " << ddo_vmm_last_error() << '\n';
            ddo_vmm_region_destroy(region);
            return EXIT_FAILURE;
        }
        if (cuMemsetD8(static_cast<CUdeviceptr>(original_address), 0x5A, ddo_vmm_region_mapped_size(region)) != CUDA_SUCCESS) {
            std::cerr << "VMM rematerialized CUDA write failed\n";
            ddo_vmm_region_destroy(region);
            return EXIT_FAILURE;
        }

        std::cout << "CUDA Driver VMM probe: PASS\n"
                  << "  virtual address: 0x" << std::hex << ddo_vmm_region_address(region) << std::dec << '\n'
                  << "  requested: " << (ddo_vmm_region_requested_size(region) / 1024.0 / 1024.0) << " MiB\n"
                  << "  mapped: " << (ddo_vmm_region_mapped_size(region) / 1024.0 / 1024.0) << " MiB\n"
                  << "  granularity: " << (ddo_vmm_device_granularity(0) / 1024.0 / 1024.0) << " MiB\n"
                  << "  evict/rematerialize at same address: PASS\n";
        ddo_vmm_region_destroy(region);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
