/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 */

// real life examples
#ifdef XTENSOR_FFTW_USE_DOUBLE
#define _USE_MATH_DEFINES  // for MSVC ("Math Constants are not defined in Standard C/C++")
#include <cmath>           // M_PI
#include <complex>

#include <xtensor-fftw/basic_double.hpp>
#include <xtensor-fftw/helper.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xbuilder.hpp>  // xt::arange
#include <xtensor/xmath.hpp>  // xt::sin, cos
#include <xtensor/xio.hpp>

#include "gtest/gtest.h"

TEST(examples, sin_derivative) {
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

  EXPECT_TRUE(xt::allclose(xt::cos(x), sin_derivative));
//  std::cout << "x:              " << x << std::endl;
//  std::cout << "sin:            " << sin << std::endl;
//  std::cout << "cos:            " << xt::cos(x) << std::endl;
//  std::cout << "sin_derivative: " << sin_derivative << std::endl;
}
#endif
