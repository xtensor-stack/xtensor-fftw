# ![xtensor-fftw](xtensor-fftw.svg)

[FFTW](http://www.fftw.org/) bindings for the [xtensor](https://github.com/xtensor-stack/xtensor) C++ multi-dimensional array library.

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/xtensor-stack/xtensor-fftw/stable?filepath=notebooks%2Fintensely_edgy_cat.ipynb)
[![Join the Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/QuantStack/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![Travis](https://travis-ci.org/xtensor-stack/xtensor-fftw.svg?branch=master)](https://travis-ci.org/xtensor-stack/xtensor-fftw)
[![Appveyor](https://ci.appveyor.com/api/projects/status/6h369haechmjeofj/branch/master?svg=true)](https://ci.appveyor.com/project/egpbos/xtensor-fftw-ivn9w/branch/master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/18861a283cf84b2e95886ba79c66e028)](https://www.codacy.com/app/egpbos/xtensor-fftw?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=egpbos/xtensor-fftw&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/egpbos/xtensor-fftw/badge.svg)](https://coveralls.io/github/egpbos/xtensor-fftw)

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/xtensor-fftw/badges/version.svg)](https://anaconda.org/conda-forge/xtensor-fftw)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/xtensor-fftw/badges/downloads.svg)](https://anaconda.org/conda-forge/xtensor-fftw)

## Introduction

_xtensor-fftw_ enables easy access to Fast Fourier Transforms (FFTs) from the [FFTW library](http://www.fftw.org/) for use on `xarray` numerical arrays from the [_xtensor_](https://github.com/xtensor-stack/xtensor) library.

Syntax and functionality are inspired by `numpy.fft`, the FFT module in the Python array programming library [NumPy](http://www.numpy.org/).

## Installation

Using `mamba` (or conda):

```bash
mamba install xtensor-fftw -c conda-forge
```

This automatically installs dependencies as well (see [list of dependencies](#dependencies) below).

Installing from source into `$PREFIX` (for instance `$CONDA_PREFIX` when in a conda environment, or `$HOME/.local`) after manually installing the [dependencies](#dependencies):

```bash
git clone https://github.com/xtensor-stack/xtensor-fftw
cd xtensor-fftw
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX
make install
```

## Dependencies

* [xtensor](https://github.com/xtensor-stack/xtensor)
* [xtl](https://github.com/xtensor-stack/xtl)
* [FFTW](http://www.fftw.org/) version 3
* A compiler supporting C++14

| `xtensor-fftw` | `xtensor`        | `xtl`   | `fftw`  |
|----------------|------------------|---------|---------|
|  master        | >=0.20.9,<0.22   |  ^0.6.9 | ^3.3.8  |
|  0.2.6         | >=0.20.9,<0.22   |  ^0.6.9 | ^3.3.8  |

## Usage

_xtensor-fftw_ is a header-only library.
To use, include one of the header files in the `include` directory, e.g. `xtensor-fftw/basic.hpp`, in your c++ code.
To compile, one should also include the paths to the FFTW header and libraries and link to the appropriate FFTW library.

FFTW allows three modes of calculus : `float`, `double` and `long double`.  
The impact of the precision type can be see below in the benchmark results.  
Use the following matrix to include, compile and link the right target:  

| `#include`                         | `precision types`          | `xtensor-fftw compile options`    | `FFTW compile options`  |
|------------------------------------|----------------------------|-----------------------------------|-------------------------|
| xtensor-fftw/basic_float.hpp       | float                      | -DXTENSOR_FFTW_USE_FLOAT=ON       | -DENABLE_FLOAT=ON       |
| xtensor-fftw/basic_double.hpp      | double                     | -DXTENSOR_FFTW_USE_DOUBLE=ON      | -DENABLE_DOUBLE=ON      |
| xtensor-fftw/basic_long_double.hpp | long double                | -DXTENSOR_FFTW_USE_LONG_DOUBLE=ON | -DENABLE_LONGDOUBLE=ON  |
| xtensor-fftw/basic_option.hpp      | depends by compile options | subset of above options           | subset of above options |
| xtensor-fftw/basic.hpp             | all types                  | no option                         | all above options       |

Specify only the required precision type to reduce the dependencies size of your application (for example for a Mobile App it matters), in fact FFTW needs to compile a specific library for each precision thus creating:
* `libfftw3f` for float precision
* `libfftw3` for double precision
* `libfftw3l` for long double precision

>__*Notes*__: FFTW allow SIMD instructions (SSE,SSE2,AVX,AVX2), OpenMP and Threads optimizations. Take a look to the availables options before compile it.  

The functions in `xtensor-fftw/basic.hpp` mimic the behavior of `numpy.fft` as much as possible.
In most cases transforms on identical input data should produce identical results within reasonable machine precision error bounds.
However, there are a few differences that one should keep in mind:

- Since FFTW expects row-major ordered arrays, _xtensor-fftw_ functions currently only accept `xarray`s with row-major layout.
By default, _xtensor_ containers use row-major layout, but take care when manually overriding this default.

- The inverse real FFT functions in FFTW destroy the input arrays during the calculation, i.e. the `irfft` family of functions in _xtensor-fftw_.
(In fact, this does not always happen, depending on which algorithm FFTW decides is most efficient in your particular situation. Don't count on it, though.)

- _xtensor-fftw_ on Windows does not support `long double` precision.
The `long double` precision version of the FFTW library requires that `sizeof(long double) == 12`.
In recent versions of Visual Studio, `long double` is an alias of `double` and has size 8.

### Example

Calculate the derivative of a (discretized) field in Fourier space, e.g. a sine shaped field `sin`:

```c++
#include <xtensor-fftw/basic.hpp>   // rfft, irfft
#include <xtensor-fftw/helper.hpp>  // rfftscale 
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>     // xt::arange
#include <xtensor/xmath.hpp>        // xt::sin, cos
#include <complex>
#include <xtensor/xio.hpp>

// generate a sinusoid field
double dx = M_PI / 100;
xt::xarray<double> x = xt::arange(0., 2 * M_PI, dx);
xt::xarray<double> sin = xt::sin(x);

// transform to Fourier space
auto sin_fs = xt::fftw::rfft(sin);

// multiply by i*k
std::complex<double> i {0, 1};
auto k = xt::fftw::rfftscale<double>(sin.shape()[0], dx);
xt::xarray<std::complex<double>> sin_derivative_fs = xt::eval(i * k * sin_fs);

// transform back to normal space
auto sin_derivative = xt::fftw::irfft(sin_derivative_fs);

std::cout << "x:              " << x << std::endl;
std::cout << "sin:            " << sin << std::endl;
std::cout << "cos:            " << xt::cos(x) << std::endl;
std::cout << "sin_derivative: " << sin_derivative << std::endl;
```

Which outputs (full output truncated):

```
x:              { 0.      ,  0.031416,  0.062832,  0.094248, ...,  6.251769}
sin:            { 0.000000e+00,  3.141076e-02,  6.279052e-02,  9.410831e-02, ..., -3.141076e-02}
cos:            { 1.000000e+00,  9.995066e-01,  9.980267e-01,  9.955620e-01, ...,  9.995066e-01}
sin_derivative: { 1.000000e+00,  9.995066e-01,  9.980267e-01,  9.955620e-01, ...,  9.995066e-01}
```

### Interactive examples
See the [notebooks folder](https://github.com/xtensor-stack/xtensor-fftw/tree/master/notebooks) for interactive Jupyter notebook examples using the C++14 [_xeus-cling_](https://github.com/jupyter-xeus/xeus-cling) kernel. These can also be run from Binder, [e.g. this one](https://mybinder.org/v2/gh/xtensor-stack/xtensor-fftw/stable?filepath=notebooks%2Fintensely_edgy_cat.ipynb).


## Building and running tests

What follows are instructions for compiling and running the _xtensor-fftw_ tests.
These also serve as an example of how to do build your own code using _xtensor-fftw_ (excluding the GoogleTest specific parts).

### Dependencies for building tests

The main dependency is a version of FFTW 3.
To enable all the precision types, FFTW must be compiled with the related flags:  
`cmake -DENABLE_FLOAT:BOOL=ON -DENABLE_LONGDOUBLE:BOOL=ON /path/of/fftw3-src`

CMake and _xtensor_ must also be installed in order to compile the _xtensor-fftw_ tests.
Both can either be installed through Conda or built/installed manually.
When using a non-Conda _xtensor_-install, make sure that the CMake `find_package` command can find _xtensor_, e.g. by passing something like `-DCMAKE_MODULE_PATH="path_to_xtensorConfig.cmake"` to CMake.
If _xtensor_ was installed in a default location, CMake should be able to find it without any command line options.

Optionally, a GoogleTest installation can be used.
However, it is recommended to use the built-in option to download GoogleTest automatically (see below).

### Configure tests

Inside the _xtensor-fftw_ source directory, create a build directory and `cd` into it:
```bash
mkdir build
cd build
```
If `pkg-config` is present on your system and your FFTW installation can be found by it, then CMake can configure your build with command:
```bash
cmake .. -DBUILD_TESTS=ON -DDOWNLOAD_GTEST=ON
```
If you do not use `pkg-config`, the FFTW prefix, i.e. the base directory under which FFTW is installed, must be passed to CMake.
Either set the `FFTWDIR` environment variable to the prefix path, or use the `FFTW_ROOT` CMake option variable.
For instance, if FFTW was installed using `./configure --prefix=/home/username/.local; make; make install`, then either set the an environment variable in your shell before running CMake:
```bash
export FFTWDIR=/home/username/.local
cmake ..  -DBUILD_TESTS=ON -DDOWNLOAD_GTEST=ON [other options]
```
or pass the path to CMake directly as such:
```bash
cmake .. -DFFTW_ROOT=/home/username/.local  -DBUILD_TESTS=ON -DDOWNLOAD_GTEST=ON [other options]
```

### Compile tests

After successful CMake configuration, run inside the build directory:
```bash
make
```

### Run tests

From the build directory, change to the test directory and run the tests:

```bash
cd test
./test_xtensor-fftw
```

## Advanced Setting

This section shows how to configure `cmake` in order to exploit advanced settings.  

### Use only Double precision

After a standard installation of FFTW library without specify a particular options, this command allow to run Test and Benchmarks using only `double` precision: 

```cmake
cmake -DBUILD_BENCHMARK=ON -DDOWNLOAD_GBENCH=ON -DBUILD_TESTS=ON -DDOWNLOAD_GTEST=ON -DFFTW_USE_FLOAT=OFF  -DFFTW_USE_LONG_DOUBLE=OFF -DFFTW_USE_DOUBLE=ON  -DCMAKE_BUILD_TYPE=Release  ..
```

Let's see what `./bench/benchmark_xtensor-fftw` produce:

```
Run on (16 X 2300 MHz CPU s)
-------------------------------------------------------------------------------
Benchmark                                        Time           CPU Iterations
-------------------------------------------------------------------------------
rfft1Dxarray_double/TransformAndInvert         66375 ns      66354 ns      10149
rfft1Dxarray_double/TransformAndInvert_nD      70856 ns      70829 ns      10128
rfft2Dxarray_double/TransformAndInvert         61264 ns      61256 ns      11456
rfft2Dxarray_double/TransformAndInvert_nD      62297 ns      62269 ns      10851
```

### Manually specify FFTW headers and link flags

This can be very useful: in this case FFTW is not required to be installed, just compiled.  
The following command produce the same results as before:  

```cmake
cmake -DBUILD_BENCHMARK=ON -DDOWNLOAD_GBENCH=ON -DBUILD_TESTS=ON -DDOWNLOAD_GTEST=ON -DFFTW_USE_FLOAT=OFF -DFFTW_USE_LONG_DOUBLE=OFF -DFFTW_USE_DOUBLE=ON -DFFTW_INCLUDE_CUSTOM_DIRS=/path/to/fftw3/api -DFFTW_LINK_FLAGS="-L/path/to/fftw3/build -lfftw3" ..
```

### Use Intel MKL

Since 2018 Intel has release a version of his famous MKL (Math Kernel Library) with a C++ and Fortran wrapper of FFTW.  
Once MKL (or oneAPI MKL) installed on the system enter the following command with adjusted path to your system:

```cmake
cmake -DBUILD_BENCHMARK=ON -DDOWNLOAD_GBENCH=ON -DBUILD_TESTS=ON -DDOWNLOAD_GTEST=ON -DFFTW_USE_FLOAT=OFF  -DFFTW_USE_LONG_DOUBLE=OFF -DFFTW_USE_DOUBLE=ON -DFFTW_INCLUDE_CUSTOM_DIRS=/opt/intel/oneapi/mkl/2021.2.0/include/fftw -DFFTW_LINK_FLAGS="-L/opt/intel/oneapi/mkl/2021.2.0/lib -L/opt/intel/oneapi/compiler/2021.2.0/mac/compiler/lib -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64 -liomp5" -DRUN_HAVE_STD_REGEX=0 -DCMAKE_BUILD_TYPE=Release ..
```

Let's see what `./bench/benchmark_xtensor-fftw` now produce:

```
Run on (16 X 2300 MHz CPU s)
-------------------------------------------------------------------------------
Benchmark                                        Time           CPU Iterations
-------------------------------------------------------------------------------
rfft1Dxarray_double/TransformAndInvert          9265 ns       9258 ns      58371
rfft1Dxarray_double/TransformAndInvert_nD       9636 ns       9602 ns      73961
rfft2Dxarray_double/TransformAndInvert         34428 ns      34427 ns      20216
rfft2Dxarray_double/TransformAndInvert_nD      37401 ns      37393 ns      19480
```

>__*Note*__: Before running test or benchmark remember to export the intel library path, e.g. on OS X: `export DYLD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2021.2.0/lib/:/opt/intel/oneapi/compiler/2021.2.0/mac/compiler/lib/`

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the [LICENSE](LICENSE) file for details.
