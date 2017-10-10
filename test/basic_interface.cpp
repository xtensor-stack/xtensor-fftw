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
#include <array>

#include <xtensor-fftw/basic.hpp>

#include "gtest/gtest.h"

///////////////////////////////////////////////////////////////////////////////
// Setup
///////////////////////////////////////////////////////////////////////////////

// GoogleTest fixture class
template <typename T>
class TransformAndInvert : public ::testing::Test {};

// the GoogleTest list of typed test cases
typedef ::testing::Types<float, double, long double> MyTypes;
TYPED_TEST_CASE(TransformAndInvert, MyTypes);

// Generates a dim-dimensional array of size n in each dimension, filled with random numbers between 0 and the numeric
// limit of type T divided by pow(n, dim) (the latter to keep the FFTs from generating infs and nans).
template <typename T, std::size_t dim>
auto generate_data(std::size_t n) {
  std::array<std::size_t, dim> shape;
  shape.fill(n);
  return xt::random::rand<T>(shape, 0, std::numeric_limits<T>::max() / std::pow(n, dim));
}

// Some testing output + the actual GoogleTest assert statement
template <typename input_t, typename fourier_t, typename output_t>
void assert_results(const input_t &a, const fourier_t &a_fourier, const output_t &should_be_a) {
  std::cout << "real input:  " << a << std::endl;
  std::cout << "fourier transform of input: " << a_fourier << std::endl;
  std::cout << "real output: " << should_be_a << std::endl;
  ASSERT_TRUE(xt::allclose(a, should_be_a));
}

// size of the randomly generated arrays along each dimension
std::size_t data_size = 4;


///////////////////////////////////////////////////////////////////////////////
// Regular FFT (complex to complex)
///////////////////////////////////////////////////////////////////////////////

////
// Regular FFT: xarray
////
/*
TYPED_TEST(TransformAndInvert, FFT_1D_xarray) {
  typedef std::complex<TypeParam> number_t;
  xt::xarray<number_t> a = generate_data<number_t, 1>(data_size);
  auto a_fourier = xt::fftw::fft(a);
  auto should_be_a = xt::fftw::ifft(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_2D_xarray) {
  typedef std::complex<TypeParam> number_t;
  xt::xarray<number_t> a = generate_data<number_t, 2>(data_size);
  auto a_fourier = xt::fftw::fft2(a);
  auto should_be_a = xt::fftw::ifft2(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_3D_xarray) {
  typedef std::complex<TypeParam> number_t;
  xt::xarray<number_t> a = generate_data<number_t, 3>(data_size);
  auto a_fourier = xt::fftw::fft3(a);
  auto should_be_a = xt::fftw::ifft3(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_4D_xarray) {
  typedef std::complex<TypeParam> number_t;
  xt::xarray<number_t> a = generate_data<number_t, 4>(data_size);
  auto a_fourier = xt::fftw::fftn(a);
  auto should_be_a = xt::fftw::ifftn(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

////
// Regular FFT: xtensor
////

TYPED_TEST(TransformAndInvert, FFT_1D_xtensor) {
  typedef std::complex<TypeParam> number_t;
  xt::xtensor<number_t, 1> a = generate_data<number_t, 1>(data_size);
  auto a_fourier = xt::fftw::fft(a);
  auto should_be_a = xt::fftw::ifft(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_2D_xtensor) {
  typedef std::complex<TypeParam> number_t;
  xt::xtensor<number_t, 2> a = generate_data<number_t, 2>(data_size);
  auto a_fourier = xt::fftw::fft2(a);
  auto should_be_a = xt::fftw::ifft2(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_3D_xtensor) {
  typedef std::complex<TypeParam> number_t;
  xt::xtensor<number_t, 3> a = generate_data<number_t, 3>(data_size);
  auto a_fourier = xt::fftw::fft3(a);
  auto should_be_a = xt::fftw::ifft3(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_4D_xtensor) {
  typedef std::complex<TypeParam> number_t;
  xt::xtensor<number_t, 4> a = generate_data<number_t, 4>(data_size);
  auto a_fourier = xt::fftw::fftn(a);
  auto should_be_a = xt::fftw::ifftn(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

*/
///////////////////////////////////////////////////////////////////////////////
// Real FFT (real input)
///////////////////////////////////////////////////////////////////////////////

////
// Real FFT: xarray
////

TYPED_TEST(TransformAndInvert, realFFT_1D_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 1>(data_size);
  auto a_fourier = xt::fftw::rfft(a);
  auto should_be_a = xt::fftw::irfft(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, realFFT_1D_xarray_fancy_templates) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 1>(data_size);
  auto a_fourier = xt::fftw::RFFT(a);
  auto should_be_a = xt::fftw::IRFFT(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}


/*
TYPED_TEST(TransformAndInvert, realFFT_2D_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 2>(data_size);
  auto a_fourier = xt::fftw::rfft2(a);
  auto should_be_a = xt::fftw::irfft2(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, realFFT_3D_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 3>(data_size);
  auto a_fourier = xt::fftw::rfft3(a);
  auto should_be_a = xt::fftw::irfft3(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, realFFT_4D_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 4>(data_size);
  auto a_fourier = xt::fftw::rfftn(a);
  auto should_be_a = xt::fftw::irfftn(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

////
// Real FFT: xtensor
////

TYPED_TEST(TransformAndInvert, realFFT_1D_xtensor) {
  xt::xtensor<TypeParam, 1> a = generate_data<TypeParam, 1>(data_size);
  auto a_fourier = xt::fftw::rfft(a);
  auto should_be_a = xt::fftw::irfft(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, realFFT_2D_xtensor) {
  xt::xtensor<TypeParam, 2> a = generate_data<TypeParam, 2>(data_size);
  auto a_fourier = xt::fftw::rfft2(a);
  auto should_be_a = xt::fftw::irfft2(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, realFFT_3D_xtensor) {
  xt::xtensor<TypeParam, 3> a = generate_data<TypeParam, 3>(data_size);
  auto a_fourier = xt::fftw::rfft3(a);
  auto should_be_a = xt::fftw::irfft3(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, realFFT_4D_xtensor) {
  xt::xtensor<TypeParam, 4> a = generate_data<TypeParam, 4>(data_size);
  auto a_fourier = xt::fftw::rfftn(a);
  auto should_be_a = xt::fftw::irfftn(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}


///////////////////////////////////////////////////////////////////////////////
// Hermitian FFT (real spectrum)
///////////////////////////////////////////////////////////////////////////////

////
// Hermitian FFT: xarray
////

TYPED_TEST(TransformAndInvert, hermFFT_1D_xarray) {
  typedef std::complex<TypeParam> number_t;
  xt::xarray<number_t> a = generate_data<number_t, 1>(data_size);
  auto a_fourier = xt::fftw::hfft(a);
  auto should_be_a = xt::fftw::ihfft(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_2D_xarray) {
  typedef std::complex<TypeParam> number_t;
  xt::xarray<number_t> a = generate_data<number_t, 2>(data_size);
  auto a_fourier = xt::fftw::hfft2(a);
  auto should_be_a = xt::fftw::ihfft2(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_3D_xarray) {
  typedef std::complex<TypeParam> number_t;
  xt::xarray<number_t> a = generate_data<number_t, 3>(data_size);
  auto a_fourier = xt::fftw::hfft3(a);
  auto should_be_a = xt::fftw::ihfft3(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_4D_xarray) {
  typedef std::complex<TypeParam> number_t;
  xt::xarray<number_t> a = generate_data<number_t, 4>(data_size);
  auto a_fourier = xt::fftw::hfftn(a);
  auto should_be_a = xt::fftw::ihfftn(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

////
// Hermitian FFT: xtensor
////

TYPED_TEST(TransformAndInvert, hermFFT_1D_xtensor) {
  typedef std::complex<TypeParam> number_t;
  xt::xtensor<number_t, 1> a = generate_data<number_t, 1>(data_size);
  auto a_fourier = xt::fftw::hfft(a);
  auto should_be_a = xt::fftw::ihfft(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_2D_xtensor) {
  typedef std::complex<TypeParam> number_t;
  xt::xtensor<number_t, 2> a = generate_data<number_t, 2>(data_size);
  auto a_fourier = xt::fftw::hfft2(a);
  auto should_be_a = xt::fftw::ihfft2(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_3D_xtensor) {
  typedef std::complex<TypeParam> number_t;
  xt::xtensor<number_t, 3> a = generate_data<number_t, 3>(data_size);
  auto a_fourier = xt::fftw::hfft3(a);
  auto should_be_a = xt::fftw::ihfft3(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_4D_xtensor) {
  typedef std::complex<TypeParam> number_t;
  xt::xtensor<number_t, 4> a = generate_data<number_t, 4>(data_size);
  auto a_fourier = xt::fftw::hfftn(a);
  auto should_be_a = xt::fftw::ihfftn(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}
*/