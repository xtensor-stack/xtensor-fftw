/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 */

#ifndef XTENSOR_FFTW_BASIC_INTERFACE_HPP
#define XTENSOR_FFTW_BASIC_INTERFACE_HPP

#include <complex>
#include <iostream>
#include <cmath> // pow
#include <array>

#include <stdexcept> // workaround for xt bug, where only including xarray does not include stdexcept; TODO: remove this include when bug is fixed!
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xcomplex.hpp>

#include "gtest/gtest.h"

#ifndef XTENSOR_FFTW_USE_FLOAT
  #define Add_Float
#else
  #if defined(XTENSOR_FFTW_USE_DOUBLE) || defined(XTENSOR_FFTW_USE_LONG_DOUBLE)
    #define Add_Float float,
  #else
    #define Add_Float float
  #endif
#endif

#ifndef XTENSOR_FFTW_USE_DOUBLE
  #define Add_Double
#else
  #if defined(XTENSOR_FFTW_USE_LONG_DOUBLE)
    #define Add_Double double,
  #else
    #define Add_Double double
  #endif
#endif

#ifndef XTENSOR_FFTW_USE_LONG_DOUBLE
  #define Add_Long_Double
#else
  #define Add_Long_Double long double
#endif

typedef ::testing::Types<Add_Float Add_Double Add_Long_Double> MyTypes;

// Generates a dim-dimensional array of size n in each dimension, filled with random numbers between 0 and the numeric
// limit of type T divided by pow(n, dim) (the latter to keep the FFTs from generating infs and nans).
template <typename T, std::size_t dim>
auto generate_data(std::size_t n) {
  std::array<std::size_t, dim> shape;
  shape.fill(n);
  return xt::random::rand<T>(shape, 0, std::numeric_limits<T>::max() / static_cast<T>(std::pow(n, dim)));
}

template <typename T, std::size_t dim>
auto generate_complex_data(std::size_t n) {
  std::complex<T> i {0,1};
  xt::xarray<std::complex<T>, xt::layout_type::row_major> c = generate_data<T, dim>(n) + generate_data<T, dim>(n) * i;
  return std::move(c) / static_cast<T>(2);  // divide by 2 (sqrt(2) would be fine too) to make sure FFT doesn't go infinite
}

template <
    typename T, std::size_t dim,
    typename xt::xarray<T> (&hfft) (const xt::xarray<std::complex<T> > &),
    typename xt::xarray<std::complex<T> > (&ihfft) (const xt::xarray<T> &)
>
auto generate_hermitian_data(std::size_t n) {
  xt::xarray<std::complex<T>, xt::layout_type::row_major> c = generate_complex_data<T, dim>(n);
  auto c_fourier = hfft(c);
  auto c_hermitian = ihfft(c_fourier);
  return std::move(c_hermitian) / static_cast<T>(10);  // divide away the FFT infinities (hopefully)
}


// Some testing output + the actual GoogleTest assert statement
template <typename input_t, typename fourier_t, typename output_t>
void assert_results(const input_t &a, const fourier_t &a_fourier, const output_t &should_be_a, bool verbose = false) {
  if (verbose) {
    std::cout << "real input:  " << a << std::endl;
    std::cout << "fourier transform of input after ifft (usually different from before): " << a_fourier << std::endl;
    std::cout << "real output: " << should_be_a << std::endl;
  }
  ASSERT_TRUE(xt::allclose(a, should_be_a, 1e-03));
}

template <typename input_t, typename fourier_t, typename output_t>
void assert_results_complex(const input_t &a, const fourier_t &a_fourier, const output_t &should_be_a, bool verbose = false) {
  if (verbose) {
    std::cout << "complex input:  " << a << std::endl;
    std::cout << "fourier transform of input after ifft (usually different from before): " << a_fourier << std::endl;
    std::cout << "complex output: " << should_be_a << std::endl;
  }
  ASSERT_TRUE(xt::allclose(xt::real(a), xt::real(should_be_a), 1e-03)
              && xt::allclose(xt::imag(a), xt::imag(should_be_a), 1e-03));
}

#endif //XTENSOR_FFTW_BASIC_INTERFACE_HPP
