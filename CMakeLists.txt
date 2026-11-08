cmake_minimum_required(VERSION 3.25)

project(ddo_vmm_allocator VERSION 0.1.0 LANGUAGES CXX)

option(DDO_VMM_BUILD_EXECUTION_OPS "Build experimental independent CUDA execution kernels" OFF)
if(DDO_VMM_BUILD_EXECUTION_OPS)
    enable_language(CUDA)
    add_library(ddo_execution_ops SHARED native/src/ddo_execution_ops.cu)
    target_include_directories(ddo_execution_ops PUBLIC native/include)
    set_target_properties(ddo_execution_ops PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    target_compile_definitions(ddo_execution_ops PRIVATE DDO_EXECUTION_OPS_BUILD=1)
    # Preserve explicit intermediate rounding; never enable fast math.
    target_compile_options(ddo_execution_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()

option(DDO_VMM_BUILD_PROBES "Build standalone CUDA Driver VMM probes" ON)
option(DDO_VMM_NATIVE_SPAN_POOL "Build the unpromoted native span-pool candidate" OFF)
option(DDO_VMM_NATIVE_JOINED_SPANS "Enable experimental dependency-preserving multi-span reservations" OFF)
if(DDO_VMM_NATIVE_JOINED_SPANS AND NOT DDO_VMM_NATIVE_SPAN_POOL)
    message(FATAL_ERROR "DDO_VMM_NATIVE_JOINED_SPANS requires DDO_VMM_NATIVE_SPAN_POOL")
endif()
option(DDO_VMM_NATIVE_SOURCE_DISPATCHER "Build the unpromoted asynchronous source dispatcher" OFF)
option(DDO_VMM_SOURCE_TEST_HOOKS "Enable isolated source lifecycle fault injection (never production)" OFF)
option(DDO_VMM_ALLOCATION_DIAGNOSTICS "Build opt-in host driver allocation timing (diagnostic only)" OFF)
option(DDO_VMM_MAPPING_TRACE "Build opt-in Windows mapping history (diagnostic only)" OFF)
if(DDO_VMM_MAPPING_TRACE AND NOT WIN32)
    message(FATAL_ERROR "DDO_VMM_MAPPING_TRACE currently requires Windows clock metadata")
endif()
option(DDO_VMM_POOL_LIFETIME_TRACE "Build bounded opt-in pool lifetime history (diagnostic only)" OFF)
option(DDO_VMM_ALLOCATION_FAILURE_TRACE "Log native pool null-return stages (diagnostic only)" OFF)
if(DDO_VMM_ALLOCATION_FAILURE_TRACE AND NOT DDO_VMM_NATIVE_SPAN_POOL)
    message(FATAL_ERROR "DDO_VMM_ALLOCATION_FAILURE_TRACE requires DDO_VMM_NATIVE_SPAN_POOL")
endif()
if(DDO_VMM_POOL_LIFETIME_TRACE AND NOT DDO_VMM_NATIVE_SPAN_POOL)
    message(FATAL_ERROR "DDO_VMM_POOL_LIFETIME_TRACE requires DDO_VMM_NATIVE_SPAN_POOL")
endif()

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
    native/src/ddo_vmm_mapping_trace.cpp
)

target_compile_features(ddo_vmm PRIVATE cxx_std_20)
target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_BUILDING_LIBRARY=1)
if(DDO_VMM_ALLOCATION_FAILURE_TRACE)
    target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_ALLOCATION_FAILURE_TRACE=1)
endif()
if(DDO_VMM_ALLOCATION_DIAGNOSTICS)
    target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_ALLOCATION_DIAGNOSTICS=1)
endif()
if(DDO_VMM_MAPPING_TRACE)
    target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_MAPPING_TRACE=1)
endif()
if(DDO_VMM_POOL_LIFETIME_TRACE)
    target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_POOL_LIFETIME_TRACE=1)
endif()
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
if(DDO_VMM_NATIVE_JOINED_SPANS)
    target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_NATIVE_JOINED_SPANS=1)
endif()
# Segmented pool owners with dead-page reuse, capacity transactions and the
# growable shared owner (cold opt-in, default off at runtime).
if(DDO_VMM_SEGMENTED_DRIVER)
    target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_SEGMENTED_DRIVER_CANDIDATE=1)
endif()
# Automatic segmented selection before CUDA initialization (cold opt-in).
if(DDO_VMM_AUTO_SEGMENTED)
    target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_AUTO_SEGMENTED_CANDIDATE=1)
endif()
# Bounded read-only timing, memory and composition observation exports.
if(DDO_VMM_TIMING_COUNTERS)
    target_compile_definitions(ddo_vmm PRIVATE DDO_VMM_TIMING_COUNTERS=1)
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
    enable_testing()
    foreach(contract execution_reclaim execution_registry)
        add_executable(ddo_vmm_${contract}_test native/tests/${contract}_test.cpp)
        target_compile_features(ddo_vmm_${contract}_test PRIVATE cxx_std_20)
        target_include_directories(ddo_vmm_${contract}_test PRIVATE native/include)
        target_link_libraries(ddo_vmm_${contract}_test PRIVATE Threads::Threads)
    endforeach()
    add_executable(ddo_vmm_owner_observation_test native/tests/owner_observation_test.cpp)
    target_compile_features(ddo_vmm_owner_observation_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_owner_observation_test PRIVATE native/include)
    add_executable(ddo_vmm_mapping_trace_test native/tests/mapping_trace_test.cpp)
    target_compile_features(ddo_vmm_mapping_trace_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_mapping_trace_test PRIVATE native/include)
    add_executable(ddo_vmm_mapping_trace_stub_test
        native/tests/mapping_trace_stub_test.cpp native/src/ddo_vmm_mapping_trace.cpp)
    target_compile_features(ddo_vmm_mapping_trace_stub_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_mapping_trace_stub_test PRIVATE native/include)
    target_compile_definitions(ddo_vmm_mapping_trace_stub_test PRIVATE DDO_VMM_BUILDING_LIBRARY=1)
    add_executable(ddo_vmm_budget_admission_test native/tests/budget_admission_test.cpp)
    target_compile_features(ddo_vmm_budget_admission_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_budget_admission_test PRIVATE native/include)
    target_compile_options(ddo_vmm_budget_admission_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_executable(ddo_vmm_host_admission_test native/tests/host_admission_test.cpp)
    target_compile_features(ddo_vmm_host_admission_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_host_admission_test PRIVATE native/src)
    target_compile_options(ddo_vmm_host_admission_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_executable(ddo_vmm_allocation_attempt_test native/tests/allocation_attempt_test.cpp)
    target_compile_features(ddo_vmm_allocation_attempt_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_allocation_attempt_test PRIVATE native/include)
    target_compile_options(ddo_vmm_allocation_attempt_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    add_executable(ddo_vmm_pool_lifetime_trace_test native/tests/pool_lifetime_trace_test.cpp)
    add_executable(ddo_vmm_allocation_failure_trace_test native/tests/allocation_failure_trace_test.cpp)
    target_compile_features(ddo_vmm_allocation_failure_trace_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_allocation_failure_trace_test PRIVATE native/include)
    target_compile_options(ddo_vmm_allocation_failure_trace_test PRIVATE
        $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
    target_compile_features(ddo_vmm_pool_lifetime_trace_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_pool_lifetime_trace_test PRIVATE native/include)
    add_executable(ddo_vmm_allocation_diagnostics_test native/tests/allocation_diagnostics_test.cpp)
    target_compile_features(ddo_vmm_allocation_diagnostics_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_allocation_diagnostics_test PRIVATE native/include)
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
    add_executable(ddo_vmm_span_joined_test native/tests/span_joined_test.cpp)
    target_compile_features(ddo_vmm_span_joined_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_span_joined_test PRIVATE native/include)
    add_executable(ddo_vmm_cache_selection_test native/tests/cache_selection_test.cpp)
    target_compile_features(ddo_vmm_cache_selection_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_cache_selection_test PRIVATE native/include)
    add_executable(ddo_vmm_allocation_footprint_test native/tests/allocation_footprint_test.cpp)
    target_compile_features(ddo_vmm_allocation_footprint_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_allocation_footprint_test PRIVATE native/include)
    add_executable(ddo_vmm_stream_dependency_test native/tests/stream_dependency_test.cpp)
    target_compile_features(ddo_vmm_stream_dependency_test PRIVATE cxx_std_20)
    target_include_directories(ddo_vmm_stream_dependency_test PRIVATE native/include)
    # Pure CPU contracts for access recovery, completed backing and pool geometry.
    foreach(contract access_recovery arena_access completed_backing completed_pool
            pool_geometry pool_geometry_capture)
        add_executable(ddo_vmm_${contract}_test native/tests/${contract}_test.cpp)
        target_compile_features(ddo_vmm_${contract}_test PRIVATE cxx_std_20)
        target_include_directories(ddo_vmm_${contract}_test PRIVATE native/include "${CUDAToolkit_INCLUDE_DIRS}")
        target_compile_options(ddo_vmm_${contract}_test PRIVATE
            $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)
        add_test(NAME ddo_vmm_${contract}_test COMMAND ddo_vmm_${contract}_test)
    endforeach()
    if(WIN32)
        add_executable(ddo_vmm_segmented_storage_cpu native/tests/segmented_storage_cpu.cpp)
        target_compile_features(ddo_vmm_segmented_storage_cpu PRIVATE cxx_std_20)
        target_include_directories(ddo_vmm_segmented_storage_cpu PRIVATE native/include "${CUDAToolkit_INCLUDE_DIRS}")
        add_test(NAME ddo_vmm_segmented_storage_cpu COMMAND ddo_vmm_segmented_storage_cpu)
        # Segmented pool contracts: the real pool source with in-process Driver
        # doubles; CUDA headers only, never a GPU library, context or device.
        foreach(contract growable composition retention timing transaction status inherited)
            add_executable(ddo_vmm_${contract}_cpu native/tests/contracts/${contract}_cpu.cpp)
            target_compile_features(ddo_vmm_${contract}_cpu PRIVATE cxx_std_20)
            target_include_directories(ddo_vmm_${contract}_cpu PRIVATE
                native/include native/tests/contracts "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDAToolkit_INCLUDE_DIRS}")
            target_compile_definitions(ddo_vmm_${contract}_cpu PRIVATE DDO_VMM_CPU_API DDO_VMM_SEGMENTED_POOL_CPU
                DDO_VMM_SEGMENTED_DRIVER_CANDIDATE DDO_VMM_NATIVE_JOINED_SPANS DDO_VMM_AUTO_SEGMENTED_CANDIDATE
                DDO_VMM_TIMING_COUNTERS)
            if(MSVC)
                target_compile_options(ddo_vmm_${contract}_cpu PRIVATE /W4 /WX /permissive-)
            endif()
            add_test(NAME ddo_vmm_${contract}_cpu COMMAND ddo_vmm_${contract}_cpu)
        endforeach()
    endif()
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
