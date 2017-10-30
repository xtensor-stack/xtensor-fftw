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

////
// Real FFT: 1D
////

template<typename precision_t>
class rfft1Dxarray : public ::benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State& state) {
      data_size = 1024;
    a = generate_data<precision_t, 1>(data_size);

    // let fftw accumulate wisdom
//    auto b = xt::fftw::rfft(a);  // DOES NOT HAVE ANY NOTICEABLE EFFECT...
//    auto c = xt::fftw::irfft(b);
  }

  void TearDown(const ::benchmark::State& /*state*/) {}

  std::size_t data_size;
  xt::xarray<precision_t> a;
};

using rfft1Dxarray_float = rfft1Dxarray<float>;

BENCHMARK_F(rfft1Dxarray_float, TransformAndInvert)(::benchmark::State& st) {
  while (st.KeepRunning()) {
    auto a_fourier = xt::fftw::rfft(a);
    ::benchmark::DoNotOptimize(a_fourier);
    auto should_be_a = xt::fftw::irfft(a_fourier);
    ::benchmark::DoNotOptimize(should_be_a);
  }
}

////
// Real FFT: nD with n = 1
////

BENCHMARK_F(rfft1Dxarray_float, TransformAndInvert_nD)(::benchmark::State& st) {
  while (st.KeepRunning()) {
    auto a_fourier = xt::fftw::rfftn<1>(a);
    ::benchmark::DoNotOptimize(a_fourier);
    auto should_be_a = xt::fftw::irfftn<1>(a_fourier);
    ::benchmark::DoNotOptimize(should_be_a);
  }
}

////
// Real FFT: 2D
////

template<typename precision_t>
class rfft2Dxarray : public ::benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State& state) {
    data_size = 64;
    a = generate_data<precision_t, 2>(data_size);

    // let fftw accumulate wisdom
//    auto b = xt::fftw::rfft(a);  // DOES NOT HAVE ANY NOTICEABLE EFFECT...
//    auto c = xt::fftw::irfft(b);
  }

  void TearDown(const ::benchmark::State& /*state*/) {}

  std::size_t data_size;
  xt::xarray<precision_t> a;
};

using rfft2Dxarray_float = rfft2Dxarray<float>;

BENCHMARK_F(rfft2Dxarray_float, TransformAndInvert)(::benchmark::State& st) {
  while (st.KeepRunning()) {
    auto a_fourier = xt::fftw::rfft2(a);
    ::benchmark::DoNotOptimize(a_fourier);
    auto should_be_a = xt::fftw::irfft2(a_fourier);
    ::benchmark::DoNotOptimize(should_be_a);
  }
}

////
// Real FFT: nD with n = 2
////

BENCHMARK_F(rfft2Dxarray_float, TransformAndInvert_nD)(::benchmark::State& st) {
  while (st.KeepRunning()) {
    auto a_fourier = xt::fftw::rfftn<2>(a);
    ::benchmark::DoNotOptimize(a_fourier);
    auto should_be_a = xt::fftw::irfftn<2>(a_fourier);
    ::benchmark::DoNotOptimize(should_be_a);
  }
}


//BENCHMARK_TEMPLATE_F(rfft1Dxarray, TransformAndInvert, double)(::benchmark::State& st) {
//  for (auto _ : st) {
//    auto a_fourier = xt::fftw::rfft(a);
//    ::benchmark::DoNotOptimize(a_fourier);
//    auto should_be_a = xt::fftw::irfft(a_fourier);
//    ::benchmark::DoNotOptimize(should_be_a);
//  }
//}
//
//#ifndef FFTW_NO_LONGDOUBLE
//BENCHMARK_TEMPLATE_F(rfft1Dxarray, TransformAndInvert, long double)(::benchmark::State& st) {
//  for (auto _ : st) {
//    auto a_fourier = xt::fftw::rfft(a);
//    ::benchmark::DoNotOptimize(a_fourier);
//    auto should_be_a = xt::fftw::irfft(a_fourier);
//    ::benchmark::DoNotOptimize(should_be_a);
//  }
//}
//#endif  // FFTW_NO_LONGDOUBLE


BENCHMARK_MAIN()
