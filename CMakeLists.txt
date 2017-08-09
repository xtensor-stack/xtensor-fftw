#
# xtensor-fftw
#
# Copyright 2017 Patrick Bos
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications:
# - Copyright 2017 Johan Mabille (rpath fix)
#

cmake_minimum_required(VERSION 3.1.3)  # 3.1.3 for set(CMAKE_CXX_STANDARD 14)

project(xtensor-fftw)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/findFFTW/")

#--------------------------------------- user options
set(FFTW_ROOT "" CACHE STRING "The FFTW prefix, i.e. the base directory under which FFTW is installed (see README.md).")
option(COMPILE_WARNINGS "Activate an list of compiler warnings. Currently for gcc and clang only." OFF)
OPTION(BUILD_TESTS "Build xtensor-fftw test suite" OFF)
#OPTION(BUILD_BENCHMARK "Build xtensor-fftw benchmarks" OFF)
OPTION(DOWNLOAD_GTEST "Build gtest from downloaded sources" OFF)
OPTION(FIX_RPATH "Correctly set rpath for the linker" OFF)
OPTION(DEFAULT_COLUMN_MAJOR "Set xtensor default layout to column major" OFF)
OPTION(COVERAGE "Enable coverage compile flags (gcc only!)" OFF)

if (COVERAGE AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    message(FATAL_ERROR "Coverage can only be used with GNU compiler!")
endif()

#--------------------------------------- build parameters for all targets
# c++ standard build options
# N.B.: these have to be set before defining targets!
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
# includes
# .. our own
set(XTENSOR_FFTW_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${XTENSOR_FFTW_INCLUDE_DIR})
# .. xtensor
find_package(xtensor REQUIRED)
include_directories(${xtensor_INCLUDE_DIRS})
# .. fftw
find_package(FFTW REQUIRED
        COMPONENTS FLOAT_LIB DOUBLE_LIB LONGDOUBLE_LIB)
include_directories(${FFTW_INCLUDE_DIRS})
# warnings (gcc and clang)
if (COMPILE_WARNINGS)
    add_compile_options(-Wundef -Wshadow -Wwrite-strings -Wredundant-decls -Woverloaded-virtual -Wcast-qual -Wcast-align -Wpointer-arith -Wconversion -Wold-style-cast -Wall -Wextra)
endif(COMPILE_WARNINGS)

#--------------------------------------- correctly set rpath for linking
if (FIX_RPATH)
    include(GNUInstallDirs)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
    if("${isSystemDir}" STREQUAL "-1")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    endif("${isSystemDir}" STREQUAL "-1")
endif(FIX_RPATH)

#--------------------------------------- install

set(XTENSOR_FFTW_HEADERS
        ${XTENSOR_FFTW_INCLUDE_DIR}/xtensor-fftw/fft.hpp
)

#--------------------------------------- tests
if(DEFAULT_COLUMN_MAJOR)
    add_definitions(-DDEFAULT_LAYOUT=layout_type::column_major)
endif()

if(DOWNLOAD_GTEST OR GTEST_SRC_DIR)
    set(BUILD_TESTS ON)
endif()

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

#if(BUILD_BENCHMARK)
#    add_subdirectory(benchmark)
#endif()