cmake_minimum_required(VERSION 3.1.3)  # 3.1.3 for set(CMAKE_CXX_STANDARD 14)

project(xtensor-fftw)

#--------------------------------------- options for all targets
# includes
# .. our own
include_directories(${CMAKE_SOURCE_DIR}/include)
# .. xtensor (TODO: make more general! Use e.g. subrepo in include or use cmake find_package.)
include_directories(/Users/pbos/sw/miniconda3/include)
# .. fftw
include_directories(/Users/pbos/sw/include)
link_directories(/Users/pbos/sw/lib)
# warnings
add_compile_options(-Wundef -Wshadow -Wwrite-strings -Wredundant-decls -Woverloaded-virtual -Wcast-qual -Wcast-align -Wpointer-arith -Wconversion -Wold-style-cast -Wall -Wextra)
# miscellaneous build options
set(CMAKE_CXX_STANDARD 14)

#--------------------------------------- tests
add_executable(test_c2c_sp_1D test/c2c_sp_1D.cpp)
target_link_libraries(test_c2c_sp_1D fftw3f)
