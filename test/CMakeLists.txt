############################################################################
# Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE-xtensor, distributed with this   #
# software.                                                                #
############################################################################
#
# Original version of this file:
# https://github.com/QuantStack/xtensor/blob/master/test/CMakeLists.txt
# commit e25330a7fe7a303c65b97ac3792a0ff934712551
#
# Modifications:
# Copyright (c) 2017, Patrick Bos (see LICENSE file at the base directory
# of the xtensor-fftw source)
#
############################################################################

cmake_minimum_required(VERSION 3.1.3)  # 3.1.3 for set(CMAKE_CXX_STANDARD 14)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xtensor-fftw-test)

    find_package(xtensor REQUIRED CONFIG)
    set(XTENSOR_INCLUDE_DIR ${xtensor_INCLUDE_DIRS})
endif ()

message(STATUS "Forcing tests build type to Release")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)

include(CheckCXXCompilerFlag)

string(TOUPPER "${CMAKE_BUILD_TYPE}" U_CMAKE_BUILD_TYPE)

set(CMAKE_CXX_STANDARD 14)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    add_compile_options(-march=native -Wunused-parameter -Wextra -Wreorder -Wconversion)
endif()

if(MSVC)
    add_compile_options(/EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
    foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
            CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        string(REPLACE "/MD" "-MT" ${flag_var} "${${flag_var}}")
    endforeach()
endif()

#if (UNIX AND NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#    add_compile_options(-stdlib=libstdc++)
#endif()

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    if(DOWNLOAD_GTEST)
        # Download and unpack googletest at configure time
        configure_file(downloadGTest.cmake.in googletest-download/CMakeLists.txt)
    else()
        # Copy local source of googletest at configure time
        configure_file(copyGTest.cmake.in googletest-download/CMakeLists.txt)
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
                    RESULT_VARIABLE result
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                     ${CMAKE_CURRENT_BINARY_DIR}/googletest-build)

    set(GTEST_INCLUDE_DIRS "${gtest_SOURCE_DIR}/include")
    set(GTEST_BOTH_LIBRARIES  gtest_main gtest)
else()
    find_package(GTest REQUIRED)
endif()

find_package(Threads)

include_directories(${XTENSOR_INCLUDE_DIR})
include_directories(${XTENSOR_FFTW_INCLUDE_DIR})
include_directories(${GTEST_INCLUDE_DIRS})

set(XTENSOR_FFTW_TESTS
        c2c_1D.cpp
)

set(XTENSOR_FFTW_TARGET test_xtensor-fftw)

add_executable(${XTENSOR_FFTW_TARGET} ${XTENSOR_FFTW_TESTS} ${XTENSOR_HEADERS} ${XTENSOR_FFTW_HEADERS})
if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    add_dependencies(${XTENSOR_FFTW_TARGET} gtest_main)
endif()
target_link_libraries(${XTENSOR_FFTW_TARGET} ${GTEST_BOTH_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}  ${FFTW_FLOAT_LIB} ${FFTW_DOUBLE_LIB} ${FFTW_LONGDOUBLE_LIB})

add_custom_target(xtest COMMAND test_xtensor-fftw DEPENDS ${XTENSOR_FFTW_TARGET})
