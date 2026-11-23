# Optional Torch bridge: a separate project, never part of the Driver-only
# ddo_vmm library build. The module is specific to one PyTorch build (version,
# git revision, CUDA variant) and one CPython ABI; the installer loads it only
# on an exact identity match recorded next to the module.
cmake_minimum_required(VERSION 3.25)
project(ddo_vmm_torch_bridge LANGUAGES CXX)

add_executable(checked_allocation_cpu checked_allocation_cpu.cpp)
target_compile_features(checked_allocation_cpu PRIVATE cxx_std_20)
target_compile_options(checked_allocation_cpu PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG /utf-8>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>)

option(DDO_VMM_BUILD_TORCH_BRIDGE "Build the Torch-linked checked allocator module" OFF)
if(DDO_VMM_BUILD_TORCH_BRIDGE)
    if(NOT MSVC OR NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(FATAL_ERROR "The Torch bridge build is reviewed for MSVC Windows x64 only")
    endif()
    foreach(root TORCH_ROOT PYTHON_ROOT CUDA_ROOT)
        if(NOT IS_DIRECTORY "${${root}}")
            message(FATAL_ERROR "Explicit ${root} directory is required")
        endif()
    endforeach()
    if(NOT PYTHON_LIBRARY)
        message(FATAL_ERROR "Explicit PYTHON_LIBRARY (for example python312.lib) is required")
    endif()
    add_library(ddo_vmm_torch_bridge MODULE ddo_vmm_torch_bridge.cpp)
    target_compile_features(ddo_vmm_torch_bridge PRIVATE cxx_std_20)
    target_compile_options(ddo_vmm_torch_bridge PRIVATE /EHsc /utf-8)
    set_property(TARGET ddo_vmm_torch_bridge PROPERTY MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    target_include_directories(ddo_vmm_torch_bridge PRIVATE
        "${TORCH_ROOT}/include" "${TORCH_ROOT}/include/torch/csrc/api/include"
        "${PYTHON_ROOT}/include" "${CUDA_ROOT}/include")
    target_link_libraries(ddo_vmm_torch_bridge PRIVATE
        "${PYTHON_LIBRARY}"
        "${TORCH_ROOT}/lib/c10.lib" "${TORCH_ROOT}/lib/c10_cuda.lib"
        "${TORCH_ROOT}/lib/torch_cpu.lib" "${TORCH_ROOT}/lib/torch_cuda.lib"
        "${TORCH_ROOT}/lib/torch_python.lib")
    set_target_properties(ddo_vmm_torch_bridge PROPERTIES PREFIX "" SUFFIX ".pyd")
endif()
