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
#include <complex>
#include <xtensor/xcomplex.hpp>

#include <xtensor-fftw/basic.hpp>

#include "gtest/gtest.h"

///////////////////////////////////////////////////////////////////////////////
// Setup
///////////////////////////////////////////////////////////////////////////////

// GoogleTest fixture class
template <typename T>
class TransformAndInvert : public ::testing::Test {};

// the GoogleTest list of typed test cases
#ifdef FFTW_NO_LONGDOUBLE
typedef ::testing::Types<float, double> MyTypes;
#else
typedef ::testing::Types<float, double, long double> MyTypes;
#endif  // FFTW_NO_LONGDOUBLE
TYPED_TEST_CASE(TransformAndInvert, MyTypes);

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
void assert_results(const input_t &a, const fourier_t &a_fourier, const output_t &should_be_a) {
  // std::cout << "real input:  " << a << std::endl;
  // std::cout << "fourier transform of input after ifft (usually different from before): " << a_fourier << std::endl;
  // std::cout << "real output: " << should_be_a << std::endl;
  ASSERT_TRUE(xt::allclose(a, should_be_a, 1e-03));
}

template <typename input_t, typename fourier_t, typename output_t>
void assert_results_complex(const input_t &a, const fourier_t &a_fourier, const output_t &should_be_a) {
  // std::cout << "complex input:  " << a << std::endl;
  // std::cout << "fourier transform of input after ifft (usually different from before): " << a_fourier << std::endl;
  // std::cout << "complex output: " << should_be_a << std::endl;
  ASSERT_TRUE(xt::allclose(xt::real(a), xt::real(should_be_a), 1e-03)
              && xt::allclose(xt::imag(a), xt::imag(should_be_a), 1e-03));
}

// size of the randomly generated arrays along each dimension
std::size_t data_size = 4;


///////////////////////////////////////////////////////////////////////////////
// Regular FFT (complex to complex)
///////////////////////////////////////////////////////////////////////////////

////
// Regular FFT: xarray
////

TYPED_TEST(TransformAndInvert, FFT_1D_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_complex_data<TypeParam, 1>(data_size);
  auto a_fourier = xt::fftw::fft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifft(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_2D_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_complex_data<TypeParam, 2>(data_size);
  auto a_fourier = xt::fftw::fft2(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifft2(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_3D_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_complex_data<TypeParam, 3>(data_size);
  auto a_fourier = xt::fftw::fft3(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifft3(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_nD_n_equals_4_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_complex_data<TypeParam, 4>(data_size);
  auto a_fourier = xt::fftw::fftn<4>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifftn<4>(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_nD_n_equals_1_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_complex_data<TypeParam, 1>(data_size);
  auto a_fourier = xt::fftw::fftn<1>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifftn<1>(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}


////
// Regular FFT: xtensor
////

/*
TYPED_TEST(TransformAndInvert, FFT_1D_xtensor) {
  xt::xtensor<std::complex<TypeParam>, 1> a = generate_complex_data<TypeParam, 1>(data_size);
  auto a_fourier = xt::fftw::fft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifft(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_2D_xtensor) {
  xt::xtensor<std::complex<TypeParam>, 2> a = generate_complex_data<TypeParam, 2>(data_size);
  auto a_fourier = xt::fftw::fft2(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifft2(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_3D_xtensor) {
  xt::xtensor<std::complex<TypeParam>, 3> a = generate_complex_data<TypeParam, 3>(data_size);
  auto a_fourier = xt::fftw::fft3(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifft3(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, FFT_4D_xtensor) {
  xt::xtensor<std::complex<TypeParam>, 4> a = generate_complex_data<TypeParam, 4>(data_size);
  auto a_fourier = xt::fftw::fftn(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ifftn(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

*/
///////////////////////////////////////////////////////////////////////////////
// Real FFT (real input)
///////////////////////////////////////////////////////////////////////////////

////
// Real FFT: xarray
////

TYPED_TEST(TransformAndInvert, realFFT_1D_xarray) {
  xt::xarray<TypeParam, xt::layout_type::row_major> a = generate_data<TypeParam, 1>(data_size);
  auto a_fourier = xt::fftw::rfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, realFFT_2D_xarray) {
  xt::xarray<TypeParam, xt::layout_type::row_major> a = generate_data<TypeParam, 2>(data_size);
  auto a_fourier = xt::fftw::rfft2(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft2(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}


TYPED_TEST(TransformAndInvert, realFFT_3D_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 3>(data_size);
  auto a_fourier = xt::fftw::rfft3(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft3(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}


TYPED_TEST(TransformAndInvert, realFFT_nD_n_equals_4_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 4>(data_size);
  auto a_fourier = xt::fftw::rfftn<4>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfftn<4>(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, realFFT_nD_n_equals_1_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 1>(data_size);
  auto a_fourier = xt::fftw::rfftn<1>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfftn<1>(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

/*
////
// Real FFT: xtensor
////

TYPED_TEST(TransformAndInvert, realFFT_1D_xtensor) {
  xt::xtensor<TypeParam, 1> a = generate_data<TypeParam, 1>(data_size);
  auto a_fourier = xt::fftw::rfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, realFFT_2D_xtensor) {
  xt::xtensor<TypeParam, 2> a = generate_data<TypeParam, 2>(data_size);
  auto a_fourier = xt::fftw::rfft2(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft2(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, realFFT_3D_xtensor) {
  xt::xtensor<TypeParam, 3> a = generate_data<TypeParam, 3>(data_size);
  auto a_fourier = xt::fftw::rfft3(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft3(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, realFFT_4D_xtensor) {
  xt::xtensor<TypeParam, 4> a = generate_data<TypeParam, 4>(data_size);
  auto a_fourier = xt::fftw::rfftn(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfftn(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}
*/

///////////////////////////////////////////////////////////////////////////////
// Hermitian FFT (real spectrum)
///////////////////////////////////////////////////////////////////////////////

////
// Hermitian FFT: xarray
////

TYPED_TEST(TransformAndInvert, hermFFT_1D_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_hermitian_data<TypeParam, 1,  xt::fftw::hfft, xt::fftw::ihfft>(data_size);
  auto a_fourier = xt::fftw::hfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_2D_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_hermitian_data<TypeParam, 2, xt::fftw::hfft2, xt::fftw::ihfft2>(data_size);
  auto a_fourier = xt::fftw::hfft2(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft2(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_3D_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_hermitian_data<TypeParam, 3, xt::fftw::hfft3, xt::fftw::ihfft3>(data_size);
  auto a_fourier = xt::fftw::hfft3(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft3(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_nD_n_equals_4_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_hermitian_data<TypeParam, 4,  xt::fftw::hfftn<4>, xt::fftw::ihfftn<4>>(data_size);
  auto a_fourier = xt::fftw::hfftn<4>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfftn<4>(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_nD_n_equals_1_xarray) {
  xt::xarray<std::complex<TypeParam>, xt::layout_type::row_major> a = generate_hermitian_data<TypeParam, 1, xt::fftw::hfft, xt::fftw::ihfft>(data_size);
  auto a_fourier = xt::fftw::hfftn<1>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfftn<1>(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

////
// Hermitian FFT: xtensor
////

/*
TYPED_TEST(TransformAndInvert, hermFFT_1D_xtensor) {
  xt::xtensor<std::complex<TypeParam>, 1> a = generate_hermitian_data<TypeParam, 1>(data_size);
  auto a_fourier = xt::fftw::hfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_2D_xtensor) {
  xt::xtensor<std::complex<TypeParam>, 2> a = generate_hermitian_data<TypeParam, 2>(data_size);
  auto a_fourier = xt::fftw::hfft2(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft2(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_3D_xtensor) {
  xt::xtensor<std::complex<TypeParam>, 3> a = generate_hermitian_data<TypeParam, 3>(data_size);
  auto a_fourier = xt::fftw::hfft3(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfft3(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert, hermFFT_4D_xtensor) {
  xt::xtensor<std::complex<TypeParam>, 4> a = generate_hermitian_data<TypeParam, 4>(data_size);
  auto a_fourier = xt::fftw::hfftn(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::ihfftn(a_fourier);
  assert_results_complex(a, a_fourier, should_be_a);
}
*/