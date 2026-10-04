cmake_minimum_required(VERSION 3.25)

project(ddo_vmm_allocator VERSION 0.1.0 LANGUAGES CXX)

option(DDO_VMM_BUILD_PROBES "Build standalone CUDA Driver VMM probes" ON)
option(DDO_VMM_NATIVE_SPAN_POOL "Build the unpromoted native span-pool candidate" OFF)
option(DDO_VMM_NATIVE_SOURCE_DISPATCHER "Build the unpromoted asynchronous source dispatcher" OFF)
option(DDO_VMM_SOURCE_TEST_HOOKS "Enable isolated source lifecycle fault injection (never production)" OFF)

# NVIDIA's Linux Python toolkit has a versioned runtime library but may lack
# the unversioned development symlink FindCUDAToolkit expects. This target uses
# only the Driver API; provide the real runtime path for toolkit discovery,
# without linking cudart or fabricating a symlink in the environment.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(CUDA_CUDART NAMES cudart libcudart.so.13 libcudart.so.12
        HINTS "${CUDAToolkit_ROOT}/lib" "$ENV{CUDAToolkit_ROOT}/lib")
endif()
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(ddo_vmm SHARED
    native/src/ddo_vmm.cpp
    native/src/ddo_vmm_source.cpp
    native/src/ddo_vmm_source_cache.cpp
    native/src/ddo_vmm_torch_allocator.cpp
    native/src/ddo_vmm_timeline.cpp
)

target_compile_features(ddo_vmm PRIVATE cxx_std_20)
target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_BUILDING_LIBRARY=1)
target_include_directories(ddo_vmm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/native/include>
)
target_link_libraries(ddo_vmm PRIVATE CUDA::cuda_driver Threads::Threads)
if(DDO_VMM_SOURCE_TEST_HOOKS)
    target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_SOURCE_TEST_HOOKS=1)
endif()
if(DDO_VMM_NATIVE_SOURCE_DISPATCHER)
    target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_NATIVE_SOURCE_DISPATCHER=1)
endif()
if(DDO_VMM_NATIVE_SPAN_POOL)
    target_sources(ddo_vmm PRIVATE native/src/ddo_vmm_torch_pool.cpp)
    target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_NATIVE_SPAN_POOL=1)
endif()

if(WIN32)
    target_link_libraries(ddo_vmm PRIVATE dxgi)
endif()

if(MSVC)
    target_compile_options(ddo_vmm PRIVATE /W4 /permissive-)
else()
    target_compile_options(ddo_vmm PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(DDO_VMM_BUILD_PROBES)
    # Pure CPU contracts: no CUDA library, context or device initialization.
    add_executable(ddo_vmm_source_flights_test native/tests/source_flights_test.cpp)
    target_compile_features(ddo_vmm_source_flights_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_source_flights_test PRIVATE native/include)
    add_executable(ddo_vmm_read_wait_test native/tests/read_wait_test.cpp)
    target_compile_features(ddo_vmm_read_wait_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_read_wait_test PRIVATE native/include)
    add_executable(ddo_vmm_source_scheduler_test native/tests/source_scheduler_test.cpp)
    target_compile_features(ddo_vmm_source_scheduler_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_source_scheduler_test PRIVATE native/include)
    add_executable(ddo_vmm_span_ledger_test native/tests/span_ledger_test.cpp)
    target_compile_features(ddo_vmm_span_ledger_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_span_ledger_test PRIVATE native/include)
    add_executable(ddo_vmm_cache_selection_test native/tests/cache_selection_test.cpp)
    target_compile_features(ddo_vmm_cache_selection_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_cache_selection_test PRIVATE native/include)
    add_executable(ddo_vmm_allocation_footprint_test native/tests/allocation_footprint_test.cpp)
    target_compile_features(ddo_vmm_allocation_footprint_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_allocation_footprint_test PRIVATE native/include)
    add_executable(ddo_vmm_stream_dependency_test native/tests/stream_dependency_test.cpp)
    target_compile_features(ddo_vmm_stream_dependency_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_stream_dependency_test PRIVATE native/include)
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
