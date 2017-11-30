/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 */

#define _USE_MATH_DEFINES  // for MSVC ("Math Constants are not defined in Standard C/C++")
#include <cmath>           // M_PI

#include <xtensor/xarray.hpp>
#include <xtensor-fftw/helper.hpp>

#include "gtest/gtest.h"


TEST(helper, fftshift) {
  xt::xarray<double> odd = {3, 4, 0, 1, 2};
  xt::xarray<double> even = {3, 4, 5, 0, 1, 2};
  xt::xarray<double> odd_range = xt::arange<double>(5);
  xt::xarray<double> even_range = xt::arange<double>(6);
  EXPECT_EQ(xt::fftw::fftshift(odd_range), odd);
  EXPECT_EQ(xt::fftw::fftshift(even_range), even);
}

TEST(helper, ifftshift) {
  xt::xarray<double> odd = {3, 4, 0, 1, 2};
  xt::xarray<double> even = {3, 4, 5, 0, 1, 2};
  EXPECT_EQ(xt::fftw::ifftshift(odd), xt::arange<double>(5));
  EXPECT_EQ(xt::fftw::ifftshift(even), xt::arange<double>(6));
}

TEST(helper, fftfreq) {
  xt::xarray<double> reference9 = {0.,  1.,  2.,  3.,  4., -4., -3., -2., -1.};
  xt::xarray<double> reference10 = {0.  ,  0.04,  0.08,  0.12,  0.16, -0.2 , -0.16, -0.12, -0.08, -0.04};
  EXPECT_EQ(xt::fftw::fftfreq(9, 1./9), reference9);
  EXPECT_EQ(xt::fftw::fftfreq(10, 2.5), reference10);
}

TEST(helper, fftscale) {
  xt::xarray<double> reference9 = {0.,  1.,  2.,  3.,  4., -4., -3., -2., -1.};
  xt::xarray<double> reference10 = {0.  ,  0.04,  0.08,  0.12,  0.16, -0.2 , -0.16, -0.12, -0.08, -0.04};
  EXPECT_EQ(xt::fftw::fftscale(9, 1./9), 2 * M_PI * reference9);
  EXPECT_EQ(xt::fftw::fftscale(10, 2.5), 2 * M_PI * reference10);
}

TEST(helper, rfftfreq) {
  xt::xarray<double> reference9 = {0.,  1.,  2.,  3.,  4.};
  xt::xarray<double> reference10 = {0.  ,  0.04,  0.08,  0.12,  0.16, 0.2};
  EXPECT_EQ(xt::fftw::rfftfreq(9, 1./9), reference9);
  EXPECT_EQ(xt::fftw::rfftfreq(10, 2.5), reference10);
}

TEST(helper, rfftscale) {
  xt::xarray<double> reference9 = {0.,  1.,  2.,  3.,  4.};
  xt::xarray<double> reference10 = {0.  ,  0.04,  0.08,  0.12,  0.16, 0.2};
  EXPECT_EQ(xt::fftw::rfftscale(9, 1./9), 2 * M_PI * reference9);
  EXPECT_EQ(xt::fftw::rfftscale(10, 2.5), 2 * M_PI * reference10);
}
