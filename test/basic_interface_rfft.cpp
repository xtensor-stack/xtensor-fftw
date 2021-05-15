/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 */

#include <xtensor-fftw/basic_option.hpp>
#include "basic_interface.hpp"

///////////////////////////////////////////////////////////////////////////////
// Setup
///////////////////////////////////////////////////////////////////////////////

// GoogleTest fixture class
template <typename T>
class TransformAndInvert_realFFT : public ::testing::Test {};

TYPED_TEST_CASE(TransformAndInvert_realFFT, MyTypes);


///////////////////////////////////////////////////////////////////////////////
// Real FFT (real input)
///////////////////////////////////////////////////////////////////////////////

////
// Real FFT: xarray
////

TYPED_TEST(TransformAndInvert_realFFT, realFFT_1D_xarray) {
  xt::xarray<TypeParam, xt::layout_type::row_major> a = generate_data<TypeParam, 1>(4);
  auto a_fourier = xt::fftw::rfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_realFFT, realFFT_2D_xarray) {
  xt::xarray<TypeParam, xt::layout_type::row_major> a = generate_data<TypeParam, 2>(4);
  auto a_fourier = xt::fftw::rfft2(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft2(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}


TYPED_TEST(TransformAndInvert_realFFT, realFFT_3D_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 3>(4);
  auto a_fourier = xt::fftw::rfft3(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft3(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}


TYPED_TEST(TransformAndInvert_realFFT, realFFT_nD_n_equals_4_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 4>(4);
  auto a_fourier = xt::fftw::rfftn<4>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfftn<4>(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_realFFT, realFFT_nD_n_equals_1_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 1>(4);
  auto a_fourier = xt::fftw::rfftn<1>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfftn<1>(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

// odd data sizes

TYPED_TEST(TransformAndInvert_realFFT, realFFT_1D_oddsize_xarray) {
  xt::xarray<TypeParam, xt::layout_type::row_major> a = generate_data<TypeParam, 1>(5);
  auto a_fourier = xt::fftw::rfft(a);
   std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft(a_fourier, true);
  assert_results(a, a_fourier, should_be_a, true);
}

TYPED_TEST(TransformAndInvert_realFFT, realFFT_2D_oddsize_xarray) {
  xt::xarray<TypeParam, xt::layout_type::row_major> a = generate_data<TypeParam, 2>(5);
  auto a_fourier = xt::fftw::rfft2(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft2(a_fourier, true);
  assert_results(a, a_fourier, should_be_a);
}


TYPED_TEST(TransformAndInvert_realFFT, realFFT_3D_oddsize_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 3>(5);
  auto a_fourier = xt::fftw::rfft3(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft3(a_fourier, true);
  assert_results(a, a_fourier, should_be_a);
}


TYPED_TEST(TransformAndInvert_realFFT, realFFT_nD_n_equals_4_oddsize_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 4>(5);
  auto a_fourier = xt::fftw::rfftn<4>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfftn<4>(a_fourier, true);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_realFFT, realFFT_nD_n_equals_1_oddsize_xarray) {
  xt::xarray<TypeParam> a = generate_data<TypeParam, 1>(5);
  auto a_fourier = xt::fftw::rfftn<1>(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfftn<1>(a_fourier, true);
  assert_results(a, a_fourier, should_be_a);
}


/*
////
// Real FFT: xtensor
////

TYPED_TEST(TransformAndInvert_realFFT, realFFT_1D_xtensor) {
  xt::xtensor<TypeParam, 1> a = generate_data<TypeParam, 1>(4);
  auto a_fourier = xt::fftw::rfft(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_realFFT, realFFT_2D_xtensor) {
  xt::xtensor<TypeParam, 2> a = generate_data<TypeParam, 2>(4);
  auto a_fourier = xt::fftw::rfft2(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft2(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_realFFT, realFFT_3D_xtensor) {
  xt::xtensor<TypeParam, 3> a = generate_data<TypeParam, 3>(4);
  auto a_fourier = xt::fftw::rfft3(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfft3(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}

TYPED_TEST(TransformAndInvert_realFFT, realFFT_4D_xtensor) {
  xt::xtensor<TypeParam, 4> a = generate_data<TypeParam, 4>(4);
  auto a_fourier = xt::fftw::rfftn(a);
  // std::cout << "fourier transform of input before ifft (which is destructive!): " << a_fourier << std::endl;
  auto should_be_a = xt::fftw::irfftn(a_fourier);
  assert_results(a, a_fourier, should_be_a);
}
*/
