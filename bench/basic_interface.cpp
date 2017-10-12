/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
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
#include <array>

#include "xtensor-fftw/basic.hpp"

#include "benchmark/benchmark.h"

// Generates a dim-dimensional array of size n in each dimension, filled with random numbers between 0 and the numeric
// limit of type T divided by pow(n, dim) (the latter to keep the FFTs from generating infs and nans).
template <typename T, std::size_t dim>
auto generate_data(std::size_t n) {
  std::array<std::size_t, dim> shape;
  shape.fill(n);
  return xt::random::rand<T>(shape, 0, std::numeric_limits<T>::max() / std::pow(n, dim));
}

template<typename precision_t>
class rfft1Dxarray : public ::benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State& state) {
    data_size = 4;
    a = generate_data<precision_t, 1>(data_size);

    // let fftw accumulate wisdom
    auto b = xt::fftw::rfft(a);
    auto c = xt::fftw::irfft(b);
  }

  void TearDown(const ::benchmark::State& /*state*/) {}

  std::size_t data_size;
  xt::xarray<precision_t> a;
};

using rfft1Dxarray_float = rfft1Dxarray<float>;

BENCHMARK_F(rfft1Dxarray_float, TransformAndInvert_OLD)(::benchmark::State& st) {
  while (st.KeepRunning()) {
    auto a_fourier = xt::fftw::rfft(a);
    ::benchmark::DoNotOptimize(a_fourier);
    auto should_be_a = xt::fftw::irfft(a_fourier);
    ::benchmark::DoNotOptimize(should_be_a);
  }
}

BENCHMARK_F(rfft1Dxarray_float, TransformAndInvert_NEW)(::benchmark::State& st) {
  while (st.KeepRunning()) {
    auto a_fourier = xt::fftw::RFFT(a);
    ::benchmark::DoNotOptimize(a_fourier);
    auto should_be_a = xt::fftw::IRFFT(a_fourier);
    ::benchmark::DoNotOptimize(should_be_a);
  }
}

BENCHMARK_MAIN()
