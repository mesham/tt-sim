cmake_minimum_required(VERSION 3.22...4.2)
project(nekbone CXX)

# Locate the TT-Metalium CMake package exported by the tt-metal build tree.
# Everything is resolved from TT_METAL_HOME (or TT_METAL_RUNTIME_ROOT, which the
# tt-sim venv sets); there are no per-machine paths in this file. Searches
# build/ then build_Release/ for the exported config and puts that build prefix
# on CMAKE_PREFIX_PATH so TT-Metalium's transitive dependencies (fmt, spdlog,
# Tracy, umd, reflect, enchantum, ...) resolve too.
if(NOT DEFINED TT_METAL_RUNTIME_ROOT)
    if(DEFINED ENV{TT_METAL_RUNTIME_ROOT})
        set(TT_METAL_RUNTIME_ROOT $ENV{TT_METAL_RUNTIME_ROOT})
    elseif(DEFINED ENV{TT_METAL_HOME})
        set(TT_METAL_RUNTIME_ROOT $ENV{TT_METAL_HOME})
    endif()
endif()
if(NOT TT-Metalium_DIR AND TT_METAL_RUNTIME_ROOT)
    foreach(_b build build_Release)
        if(EXISTS "${TT_METAL_RUNTIME_ROOT}/${_b}/lib/cmake/tt-metalium/tt-metalium-config.cmake")
            set(TT-Metalium_DIR "${TT_METAL_RUNTIME_ROOT}/${_b}/lib/cmake/tt-metalium")
            list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_RUNTIME_ROOT}/${_b}")
            break()
        endif()
    endforeach()
endif()

add_executable(nekbone nekbone.cpp)
find_package(TT-Metalium REQUIRED)
target_link_libraries(nekbone PRIVATE TT::Metalium)
target_include_directories(nekbone PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# MeshBuffer::address() returns DeviceAddr (uint64_t) and SetRuntimeArgs takes
# uint32_t; the brace-init lists in nekbone.cpp narrow. The 0.65 Makefile
# silenced exactly this with clang's -Wno-c++11-narrowing. Silenced here in the
# build rather than in the source, so nekbone.cpp stays byte-identical to 0.65.
target_compile_options(nekbone PRIVATE -Wno-narrowing)
