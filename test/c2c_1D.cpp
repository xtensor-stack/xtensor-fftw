/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 */

#include <stdexcept> // workaround for xt bug, where only including xarray does not include stdexcept; TODO: remove this include when bug is fixed!
#include <xtensor/xarray.hpp>
#include <iostream>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include <xtensor-fftw/fft.hpp>

#include "gtest/gtest.h"

template <typename T>
class fftC2C1D : public ::testing::Test {};

typedef ::testing::Types<float, double, long double> MyTypes;
TYPED_TEST_CASE(fftC2C1D, MyTypes);

TYPED_TEST(fftC2C1D, TransformAndInvert) {
  xt::xarray<TypeParam> a = xt::random::rand<TypeParam>({8}, 0, std::numeric_limits<TypeParam>::max()/8);

  auto a_fourier = xt::fftw::fft(a);

  auto should_be_a = xt::fftw::ifft(a_fourier);

  std::cout << "real input:  " << a << std::endl;
  std::cout << "fourier transform of input: " << a_fourier << std::endl;
  std::cout << "real output: " << should_be_a << std::endl;
  ASSERT_TRUE(xt::allclose(a, should_be_a));
}