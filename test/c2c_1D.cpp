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
#include <cmath> // pow

#include <xtensor-fftw/fft.hpp>

#include "gtest/gtest.h"

template <typename T>
class TransformAndInvert : public ::testing::Test {};

// Define a class that holds the two types we want to pass to our GoogleTest typed tests
// (https://stackoverflow.com/a/29382470/1199693). The variadic template parameter `typename...`
// is because `xarray` and `xtensor` have more template parameters than we care about here.
template <template<typename /* precision */, std::size_t /* dim */, typename...> class _container, typename _precision>
struct TDefs {
  template<typename T, std::size_t dim> using container = _container<T, dim>;
  using precision = _precision;
};

// make a uniform template interface for xarray and xtensor (where dim does nothing for xarray)
template<typename precision, std::size_t dim> using xarray_front = xt::xarray<precision>;
template<typename precision, std::size_t dim> using xtensor_front = xt::xtensor<precision, dim>;

// the GoogleTest list of typed test cases
typedef ::testing::Types<
    TDefs<xarray_front, float>,
    TDefs<xarray_front, double>,
    TDefs<xarray_front, long double>,
    TDefs<xtensor_front, float>,
    TDefs<xtensor_front, double>,
    TDefs<xtensor_front, long double>
  > MyTypes;
TYPED_TEST_CASE(TransformAndInvert, MyTypes);

// forward declaration (definition below)
template <typename T, size_t dim> auto generate_data(size_t n);
std::size_t data_size = 4;

TYPED_TEST(TransformAndInvert, R2C2R_1D) {
  std::size_t dim = 1;

  using precision = typename TypeParam::precision;
  template <typename real_t, std::size_t dim> using container = class TypeParam::template container<real_t, dim>;
//using container = typename TypeParam::template container;
  // actual test logic
  container<precision, dim> a = generate_data<precision, dim>(data_size);

  auto a_fourier = xt::fftw::fft(a);

  auto should_be_a = xt::fftw::ifft(a_fourier);

  std::cout << "real input:  " << a << std::endl;
  std::cout << "fourier transform of input: " << a_fourier << std::endl;
  std::cout << "real output: " << should_be_a << std::endl;
  ASSERT_TRUE(xt::allclose(a, should_be_a));
}


// Generates a dim-dimensional array of size n in each dimension, filled with random numbers between 0 and the numeric
// limit of type T divided by pow(n, dim) (the latter to keep the FFTs from generating infs and nans).
template <typename T, std::size_t dim>
auto generate_data(std::size_t n) {
  return xt::random::rand<T>(std::array<std::size_t, dim>().fill(n),
                             0, std::numeric_limits<T>::max() / std::pow(n, dim));
}
