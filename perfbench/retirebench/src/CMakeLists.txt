cmake_minimum_required(VERSION 3.22...4.2)
project(retirebench CXX)

# Locate the TT-Metalium CMake package exported by the tt-metal build tree.
# Honours TT_METAL_RUNTIME_ROOT (set by the tt-sim venv), falling back to the
# older TT_METAL_HOME. Searches build/ then build_Release/ for the exported
# config, and puts that build prefix on CMAKE_PREFIX_PATH so TT-Metalium's
# transitive dependencies (fmt, spdlog, Tracy, umd, ...) resolve too.
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

add_executable(retirebench retirebench.cpp)
find_package(TT-Metalium REQUIRED)
target_link_libraries(retirebench PRIVATE TT::Metalium)
