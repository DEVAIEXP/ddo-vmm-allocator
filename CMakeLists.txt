cmake_minimum_required(VERSION 3.25)

project(ddo_vmm_allocator VERSION 0.1.0 LANGUAGES CXX)

option(DDO_VMM_BUILD_PROBES "Build standalone CUDA Driver VMM probes" ON)

find_package(CUDAToolkit REQUIRED)

add_library(ddo_vmm SHARED
    native/src/ddo_vmm.cpp
    native/src/ddo_vmm_torch_allocator.cpp
)

target_compile_features(ddo_vmm PRIVATE cxx_std_20)
target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_BUILDING_LIBRARY=1)
target_include_directories(ddo_vmm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/native/include>
)
target_link_libraries(ddo_vmm PRIVATE CUDA::cuda_driver)

if(MSVC)
    target_compile_options(ddo_vmm PRIVATE /W4 /permissive-)
else()
    target_compile_options(ddo_vmm PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(DDO_VMM_BUILD_PROBES)
    add_executable(ddo_vmm_driver_probe native/tests/vmm_driver_probe.cpp)
    target_compile_features(ddo_vmm_driver_probe PRIVATE cxx_std_20)
    target_link_libraries(ddo_vmm_driver_probe PRIVATE ddo_vmm CUDA::cuda_driver)
    add_test(NAME ddo_vmm_driver_probe COMMAND ddo_vmm_driver_probe)
endif()

# CUDA's Windows Driver import library is built against the static CRT.  Keep
# this standalone C ABI library on that CRT boundary; a future PyTorch adapter
# will be a separate target with the runtime policy required by PyTorch.
if(MSVC)
    set_property(TARGET ddo_vmm PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    if(TARGET ddo_vmm_driver_probe)
        set_property(TARGET ddo_vmm_driver_probe PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()
