cmake_minimum_required(VERSION 3.22...4.2)
project(six CXX)

# Locate the TT-Metalium CMake package exported by the tt-metal build tree.
# Honours TT_METAL_HOME, falling back to TT_METAL_RUNTIME_ROOT (set by the
# tt-sim venv). Searches build/ then build_Release/ for the exported config,
# and puts that build prefix on CMAKE_PREFIX_PATH so TT-Metalium's transitive
# dependencies (fmt, spdlog, Tracy, umd, ...) resolve too.
if(NOT DEFINED TT_METAL_HOME)
    if(DEFINED ENV{TT_METAL_HOME})
        set(TT_METAL_HOME $ENV{TT_METAL_HOME})
    elseif(DEFINED ENV{TT_METAL_RUNTIME_ROOT})
        set(TT_METAL_HOME $ENV{TT_METAL_RUNTIME_ROOT})
    endif()
endif()
if(NOT TT-Metalium_DIR AND TT_METAL_HOME)
    foreach(_b build build_Release)
        if(EXISTS "${TT_METAL_HOME}/${_b}/lib/cmake/tt-metalium/tt-metalium-config.cmake")
            set(TT-Metalium_DIR "${TT_METAL_HOME}/${_b}/lib/cmake/tt-metalium")
            list(APPEND CMAKE_PREFIX_PATH "${TT_METAL_HOME}/${_b}")
            break()
        endif()
    endforeach()
endif()

add_executable(six six.cpp)
find_package(TT-Metalium REQUIRED)
target_link_libraries(six PRIVATE TT::Metalium)
