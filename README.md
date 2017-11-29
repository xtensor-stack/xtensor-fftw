# ![xtensor-fftw](http://quantstack.net/assets/images/xtensor-fftw.svg)

[![Travis](https://travis-ci.org/egpbos/xtensor-fftw.svg?branch=master)](https://travis-ci.org/egpbos/xtensor-fftw)
[![Appveyor](https://ci.appveyor.com/api/projects/status/l4wgk98kwospu7n1?svg=true)](https://ci.appveyor.com/project/egpbos/xtensor-fftw)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/18861a283cf84b2e95886ba79c66e028)](https://www.codacy.com/app/egpbos/xtensor-fftw?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=egpbos/xtensor-fftw&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/egpbos/xtensor-fftw/badge.svg)](https://coveralls.io/github/egpbos/xtensor-fftw)
[![Join the Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/QuantStack/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[fftw](http://www.fftw.org/) bindings for the [xtensor](https://github.com/QuantStack/xtensor) C++ multi-dimensional array library.

## Example

Calculate the derivative of a field `a` in Fourier space, e.g. a sine shaped field:

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

## Usage

_xtensor-fftw_ is a header-only library.
To use, include one of the header files in the `include` directory, e.g. `xtensor-fftw/fft.hpp`, in your c++ code.
To compile, one should also include the paths to the FFTW header and libraries and link to the appropriate FFTW library.

Note that _xtensor-fftw_ on Windows does not support `long double` precision.
The `long double` precision version of the FFTW library requires that `sizeof(long double) == 12`.
In recent versions of Visual Studio, `long double` is an alias of `double` and has size 8.

What follows are instructions for compiling the _xtensor-fftw_ tests.
These also serve as an example of how to do build your own code using _xtensor-fftw_ (excluding the GoogleTest specific parts).

### Dependencies
The main dependency is a version of FFTW 3.
For the tests, we need the floating point version which is enabled in the FFTW configuration step using:
```bash
./configure --enable-float
```

CMake and _xtensor_ must also be installed in order to compile the _xtensor-fftw_ tests.
Both can either be installed through Conda or built/installed manually.
When using a non-Conda _xtensor_-install, make sure that the CMake `find_package` command can find _xtensor_, e.g. by passing something like `-DCMAKE_MODULE_PATH="path_to_xtensorConfig.cmake"` to CMake (not tested).

Optionally, a GoogleTest installation can be used.
However, it is recommended to use the built-in option to download GoogleTest automatically (see below).

### Configure

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

### Compile

After successful CMake configuration, run inside the build directory:
```bash
make
```

## License

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the [LICENSE](LICENSE) file for details.
