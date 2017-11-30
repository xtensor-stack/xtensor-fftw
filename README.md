# ![xtensor-fftw](http://quantstack.net/assets/images/xtensor-fftw.svg)

[![Travis](https://travis-ci.org/egpbos/xtensor-fftw.svg?branch=master)](https://travis-ci.org/egpbos/xtensor-fftw)
[![Appveyor](https://ci.appveyor.com/api/projects/status/l4wgk98kwospu7n1?svg=true)](https://ci.appveyor.com/project/egpbos/xtensor-fftw)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/18861a283cf84b2e95886ba79c66e028)](https://www.codacy.com/app/egpbos/xtensor-fftw?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=egpbos/xtensor-fftw&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/egpbos/xtensor-fftw/badge.svg)](https://coveralls.io/github/egpbos/xtensor-fftw)
[![Join the Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/QuantStack/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[FFTW](http://www.fftw.org/) bindings for the [xtensor](https://github.com/QuantStack/xtensor) C++ multi-dimensional array library.

## Introduction

_xtensor-fftw_ enables easy access to Fast Fourier Transforms (FFTs) from the [FFTW library](http://www.fftw.org/) for use on `xarray` numerical arrays from the [_xtensor_](https://github.com/QuantStack/xtensor) library.

Syntax and functionality are inspired by `numpy.fft`, the FFT module in the Python array programming library [NumPy](http://www.numpy.org/).

## Installation

Using `conda`:

```bash
conda install xtensor-fftw -c conda-forge
```

This automatically installs dependencies as well (see [list of dependencies](#dependencies) below).

Installing from source into `$PREFIX` (for instance `$CONDA_PREFIX` when in a conda environment, or `$HOME/.local`) after manually installing the [dependencies](#dependencies):

```bash
git clone https://github.com/egpbos/xtensor-fftw
cd xtensor-fftw
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX
make install
```

## Dependencies

* [xtensor](https://github.com/QuantStack/xtensor)
* [xtl](https://github.com/QuantStack/xtl)
* [FFTW](http://www.fftw.org/) version 3
* A compiler supporting C++14

## Usage

_xtensor-fftw_ is a header-only library.
To use, include one of the header files in the `include` directory, e.g. `xtensor-fftw/basic.hpp`, in your c++ code.
To compile, one should also include the paths to the FFTW header and libraries and link to the appropriate FFTW library.

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
double dx = M_PI/100;
xt::xarray<double> x = xt::arange(0., 2*M_PI, dx);
xt::xarray<double> sin = xt::sin(x);

// transform to Fourier space
auto sin_fs = xt::fftw::rfft(sin);

// multiply by i*k
std::complex<double> i {0, 1};
auto k = xt::fftw::rfftscale<double>(sin.shape()[0], dx);
xt::xarray< std::complex<double> > sin_derivative_fs = xt::eval(i * k * sin_fs);

// transform back to normal space
auto sin_derivative = xt::fftw::irfft(sin_derivative_fs);

std::cout << "x:              " << x << std::endl;
std::cout << "sin:            " << sin << std::endl;
std::cout << "cos:            " << xt::cos(x) << std::endl;
std::cout << "sin_derivative: " << sin_derivative << std::endl;
```


## Building and running tests

What follows are instructions for compiling and running the _xtensor-fftw_ tests.
These also serve as an example of how to do build your own code using _xtensor-fftw_ (excluding the GoogleTest specific parts).

### Dependencies for building tests
The main dependency is a version of FFTW 3.
For the tests, we need the floating point version which is enabled in the FFTW configuration step using:
```bash
./configure --enable-float
```

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

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the [LICENSE](LICENSE) file for details.
