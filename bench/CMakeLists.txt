############################################################################
# Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################
#
# Original version of this file:
# https://github.com/xtensor-stack/xtensor/blob/master/test/CMakeLists.txt
# commit e25330a7fe7a303c65b97ac3792a0ff934712551
#
# Adaptation from GoogleTest to Google Benchmark:
# Copyright (c) 2017, Patrick Bos
#
############################################################################

cmake_minimum_required(VERSION 3.1.3)  # 3.1.3 for set(CMAKE_CXX_STANDARD 14)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xtensor-fftw-bench)

    find_package(xtensor REQUIRED CONFIG)
    set(XTENSOR_INCLUDE_DIR ${xtensor_INCLUDE_DIRS})
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Building benchmarks in Debug mode will give suboptimal performance.")
endif ()

include(CheckCXXCompilerFlag)

#string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

set(CMAKE_CXX_STANDARD 14)

#if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
#    add_compile_options(-march=native) #-Wunused-parameter -Wextra -Wreorder -Wconversion)
#endif()
#
#if(MSVC)
#    add_compile_options(/EHsc /MP /bigobj)
#    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
#    foreach(flag_var
#            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
#            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
#        string(REPLACE "/MD" "-MT" ${flag_var} "${${flag_var}}")
#    endforeach()
#endif()

#if (UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#    add_compile_options(-stdlib=libstdc++)
#endif()

if(DOWNLOAD_GBENCH OR GBENCH_SRC_DIR)
    if(DOWNLOAD_GBENCH)
        # Download and unpack google benchmark at configure time
        configure_file(downloadGBench.cmake.in googlebench-download/CMakeLists.txt)
    else()
        # Copy local source of google benchmark at configure time
        configure_file(copyGBench.cmake.in googlebench-download/CMakeLists.txt)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googlebench-download )
    if(result)
        message(FATAL_ERROR "CMake step for googlebench failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googlebench-download )
    if(result)
        message(FATAL_ERROR "Build step for googlebench failed: ${result}")
    endif()

    # Add google benchmark directly to our build. This defines
    # the benchmark target.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googlebench-src
            ${CMAKE_CURRENT_BINARY_DIR}/googlebench-build)

    set(benchmark_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/googlebench-src/include")
    set(benchmark_BOTH_LIBRARIES  benchmark)
else()
    find_package(benchmark REQUIRED)
endif()

find_package(Threads)

include_directories(${XTENSOR_INCLUDE_DIR})
include_directories(${xtl_INCLUDE_DIRS})
include_directories(${XTENSOR_FFTW_INCLUDE_DIR})
include_directories(${benchmark_INCLUDE_DIRS})

set(XTENSOR_FFTW_BENCHMARKS
        basic_interface.cpp
        )

set(XTENSOR_FFTW_TARGET benchmark_xtensor-fftw)

add_executable(${XTENSOR_FFTW_TARGET} ${XTENSOR_FFTW_BENCHMARKS} ${XTENSOR_HEADERS} ${XTENSOR_FFTW_HEADERS})
if(DOWNLOAD_GBENCH OR GBENCH_SRC_DIR)
    add_dependencies(${XTENSOR_FFTW_TARGET} benchmark)
endif()
target_link_libraries(${XTENSOR_FFTW_TARGET} ${benchmark_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${FFTW_LINK_FLAGS})

add_custom_target(xbench COMMAND benchmark_xtensor-fftw DEPENDS ${XTENSOR_FFTW_TARGET})
