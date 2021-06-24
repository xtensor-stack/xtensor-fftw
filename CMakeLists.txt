#
# xtensor-fftw
# Copyright (c) 2017, Patrick Bos
#
# Distributed under the terms of the BSD 3-Clause License.
#
# The full license is in the file LICENSE, distributed with this software.
#
# Modifications:
# - Copyright 2017 Johan Mabille (rpath fix)
#

cmake_minimum_required(VERSION 3.1.3)  # 3.1.3 for set(CMAKE_CXX_STANDARD 14)

project(xtensor-fftw)

set(XTENSOR_FFTW_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(FFTW_INCLUDE_CUSTOM_DIRS "" CACHE STRING "Set the FFTW include dir without the requirement of FFTW installation.")
set(FFTW_LINK_FLAGS "" CACHE STRING "Set the CXX library to link, e.g.: -L/usr/local -lfftw3")
#--------------------------------------- versioning

file(STRINGS "${XTENSOR_FFTW_INCLUDE_DIR}/xtensor-fftw/xtensor-fftw_config.hpp" xtensor-fftw_version_defines
        REGEX "#define XTENSOR_FFTW_VERSION_(MAJOR|MINOR|PATCH)")
foreach(ver ${xtensor-fftw_version_defines})
    if(ver MATCHES "#define XTENSOR_FFTW_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(XTENSOR_FFTW_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()
set(${PROJECT_NAME}_VERSION
        ${XTENSOR_FFTW_VERSION_MAJOR}.${XTENSOR_FFTW_VERSION_MINOR}.${XTENSOR_FFTW_VERSION_PATCH})
message(STATUS "Building xtensor-fftw v${${PROJECT_NAME}_VERSION}")

#--------------------------------------- cmake modules

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/findFFTW/")

#--------------------------------------- user options
set(FFTW_ROOT "" CACHE STRING "The FFTW prefix, i.e. the base directory under which FFTW is installed (see README.md).")
option(COMPILE_WARNINGS "Activate an list of compiler warnings. Currently for gcc and clang only." OFF)
OPTION(BUILD_TESTS "Build xtensor-fftw test suite" OFF)
OPTION(DOWNLOAD_GTEST "Build gtest from downloaded sources" OFF)
OPTION(BUILD_BENCHMARK "Build xtensor-fftw benchmarks" OFF)
OPTION(DOWNLOAD_GBENCH "Build Google Benchmark from downloaded sources" OFF)
OPTION(BENCHMARK_ENABLE_TESTING "Build tests for Google Benchmark" OFF)
OPTION(FIX_RPATH "Correctly set rpath for the linker" OFF)
OPTION(DEFAULT_COLUMN_MAJOR "Set xtensor default layout to column major. This is currently not supported, since FFTW demands row major layout." OFF)
OPTION(COVERAGE "Enable coverage compile flags (gcc only!)" OFF)
OPTION(DISABLE_EXCEPTIONS "Disable C++ exceptions" OFF)
OPTION(FFTW_USE_FLOAT "Enable FFTW Float type" ON)
OPTION(FFTW_USE_DOUBLE "Enable FFTW Double type" ON)
OPTION(FFTW_USE_LONG_DOUBLE "Enable FFTW Long Double type" ON)

if(FFTW_USE_FLOAT)
    add_definitions(-DXTENSOR_FFTW_USE_FLOAT)
    set(REQUIRE_FLOAT_LIB "FLOAT_LIB")
else()
    set(REQUIRE_FLOAT_LIB "")
endif()

if(FFTW_USE_DOUBLE)
    add_definitions(-DXTENSOR_FFTW_USE_DOUBLE)
    set(REQUIRE_DOUBLE_LIB "DOUBLE_LIB")
else()
    set(REQUIRE_DOUBLE_LIB "")
endif()

if(FFTW_USE_LONG_DOUBLE AND NOT MSVC)
    add_definitions(-DXTENSOR_FFTW_USE_LONG_DOUBLE)
    set(REQUIRE_LONG_DOUBLE_LIB "LONGDOUBLE_LIB")
else()
    set(REQUIRE_LONG_DOUBLE_LIB "")
    set(FFTW_USE_LONG_DOUBLE OFF)
endif()

if(NOT REQUIRE_FLOAT_LIB AND NOT REQUIRE_DOUBLE_LIB AND NOT REQUIRE_LONG_DOUBLE_LIB) 
    message(FATAL_ERROR "Please, select at least one of the available FFTW type libraries")
endif()

if (COVERAGE)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        message(FATAL_ERROR "Coverage can only be used with GNU and Clang compilers!")
    endif()
endif()

#--------------------------------------- build parameters for all targets
# c++ standard build options
# N.B.: these have to be set before defining targets!
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# includes

# .. our own
include_directories(${XTENSOR_FFTW_INCLUDE_DIR})

# .. xtl
set(xtl_REQUIRED_VERSION 0.6.9)
if(TARGET xtl)
    set(xtl_VERSION ${XTL_VERSION_MAJOR}.${XTL_VERSION_MINOR}.${XTL_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if( NOT ${xtl_VERSION} VERSION_GREATER_EQUAL ${xtl_REQUIRED_VERSION})
        message(ERROR "Mismatch xtl versions. Found '${xtl_VERSION}' but requires: '${xtl_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xtl v${xtl_VERSION}")
    endif()
else()
    find_package(xtl ${xtl_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xtl: ${xtl_INCLUDE_DIRS}/xtl")
endif()

# .. xtensor
set(xtensor_REQUIRED_VERSION 0.20.9)
if(TARGET xtensor)
    set(xtensor_VERSION ${XTENSOR_VERSION_MAJOR}.${XTENSOR_VERSION_MINOR}.${XTENSOR_VERSION_PATCH})
    # Note: This is not SEMVER compatible comparison
    if( NOT ${xtensor_VERSION} VERSION_GREATER_EQUAL ${xtensor_REQUIRED_VERSION})
        message(ERROR "Mismatch xtensor versions. Found '${xtensor_VERSION}' but requires: '${xtensor_REQUIRED_VERSION}'")
    else()
        message(STATUS "Found xtensor v${xtensor_VERSION}")
    endif()
else()
    find_package(xtensor ${xtensor_REQUIRED_VERSION} REQUIRED)
    message(STATUS "Found xtensor: ${xtensor_INCLUDE_DIRS}/xtensor")
endif()

# .. fftw
if(NOT "${FFTW_INCLUDE_CUSTOM_DIRS}" STREQUAL "")
    include_directories(${FFTW_INCLUDE_CUSTOM_DIRS})
else()
    if(MSVC)
        # no long double component, since in the Windows conda-forge build it is not available
        # and the "official" prebuilt long double library can only be used from MinGW
        find_package(FFTW REQUIRED
                COMPONENTS ${REQUIRE_FLOAT_LIB} ${REQUIRE_DOUBLE_LIB})
        add_definitions(-DFFTW_NO_LONGDOUBLE)
    else(MSVC)
        find_package(FFTW REQUIRED
                COMPONENTS ${REQUIRE_FLOAT_LIB} ${REQUIRE_DOUBLE_LIB} ${REQUIRE_LONG_DOUBLE_LIB})
    endif()
    include_directories(${FFTW_INCLUDE_DIRS})

    # link only with selected libraries
    set(FFTW_LINK_FLAGS "")
    if(FFTW_USE_FLOAT)
        set(FFTW_LINK_FLAGS ${FFTW_FLOAT_LIB})
    endif()

    if(FFTW_USE_DOUBLE)
        set(FFTW_LINK_FLAGS ${FFTW_LINK_FLAGS} ${FFTW_DOUBLE_LIB})
    endif()

    if(FFTW_USE_LONG_DOUBLE)
        set(FFTW_LINK_FLAGS ${FFTW_LINK_FLAGS} ${FFTW_LONGDOUBLE_LIB})
    endif()
endif()

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

#--------------------------------------- library contents

set(XTENSOR_FFTW_HEADERS
        ${XTENSOR_FFTW_INCLUDE_DIR}/xtensor-fftw/basic_double.hpp
        ${XTENSOR_FFTW_INCLUDE_DIR}/xtensor-fftw/basic_float.hpp
        ${XTENSOR_FFTW_INCLUDE_DIR}/xtensor-fftw/basic_long_double.hpp
        ${XTENSOR_FFTW_INCLUDE_DIR}/xtensor-fftw/basic_option.hpp
        ${XTENSOR_FFTW_INCLUDE_DIR}/xtensor-fftw/basic.hpp
        ${XTENSOR_FFTW_INCLUDE_DIR}/xtensor-fftw/common.hpp
        ${XTENSOR_FFTW_INCLUDE_DIR}/xtensor-fftw/helper.hpp
        ${XTENSOR_FFTW_INCLUDE_DIR}/xtensor-fftw/xtensor-fftw_config.hpp
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

if(BUILD_BENCHMARK)
    add_subdirectory(bench)
endif()


#--------------------------------------- installation

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(FILES ${XTENSOR_FFTW_HEADERS}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xtensor-fftw)

set(XTENSOR_FFTW_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}" CACHE
        STRING "install path for xtensor-fftwConfig.cmake")

configure_package_config_file(${PROJECT_NAME}Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
        INSTALL_DESTINATION ${XTENSOR_FFTW_CMAKECONFIG_INSTALL_DIR})

# xtensor-fftw is header-only and does not depend on the architecture.
# Remove CMAKE_SIZEOF_VOID_P from xtensor-fftwConfigVersion.cmake so that an xtensor-fftwConfig.cmake
# generated for a 64 bit target can be used for 32 bit targets and vice versa.
set(_XTENSOR_FFTW_CMAKE_SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})
unset(CMAKE_SIZEOF_VOID_P)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        VERSION ${${PROJECT_NAME}_VERSION}
        COMPATIBILITY AnyNewerVersion)
set(CMAKE_SIZEOF_VOID_P ${_XTENSOR_FFTW_CMAKE_SIZEOF_VOID_P})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION ${XTENSOR_FFTW_CMAKECONFIG_INSTALL_DIR})

configure_file(${PROJECT_NAME}.pc.in
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")
