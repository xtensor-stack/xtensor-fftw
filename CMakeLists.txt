cmake_minimum_required(VERSION 3.1.3)  # 3.1.3 for set(CMAKE_CXX_STANDARD 14)

project(xtensor-fftw)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/findFFTW/")

#--------------------------------------- user options
set(FFTW_ROOT "" CACHE STRING "The FFTW prefix, i.e. the base directory under which FFTW is installed (see README.md).")

#--------------------------------------- build parameters for all targets
# includes
# .. our own
include_directories(${CMAKE_SOURCE_DIR}/include)
find_package(xtensor REQUIRED)
include_directories(${xtensor_INCLUDE_DIRS})
# .. fftw
find_package(FFTW REQUIRED
        COMPONENTS FLOAT_LIB)
include_directories(${FFTW_INCLUDE_DIRS})
# warnings
add_compile_options(-Wundef -Wshadow -Wwrite-strings -Wredundant-decls -Woverloaded-virtual -Wcast-qual -Wcast-align -Wpointer-arith -Wconversion -Wold-style-cast -Wall -Wextra)
# miscellaneous build options
set(CMAKE_CXX_STANDARD 14)

#--------------------------------------- tests
add_executable(test_c2c_sp_1D test/c2c_sp_1D.cpp)
target_link_libraries(test_c2c_sp_1D ${FFTW_FLOAT_LIB})
