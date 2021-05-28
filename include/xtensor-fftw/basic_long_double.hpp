/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * basic_long_double.hpp:
 * Contains the basic functions needed to do FFTs and inverse FFTs on long double 
 * and complex<long double> arrays. The behavior of these functions mimics that of
 * the numpy.fft module, see https://github.com/xtensor-stack/xtensor-fftw/issues/6.
 *
 */

#ifndef XTENSOR_FFTW_BASIC_LONG_DOUBLE_HPP
#define XTENSOR_FFTW_BASIC_LONG_DOUBLE_HPP

#include "common.hpp"

namespace xt {
  namespace fftw {

    template <> struct fftw_t<long double> {
      using plan = fftwl_plan;
      using complex = fftwl_complex;
      constexpr static void (&execute)(plan) = fftwl_execute;
      constexpr static void (&destroy_plan)(plan) = fftwl_destroy_plan;
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Regular FFT (complex to complex)
    ///////////////////////////////////////////////////////////////////////////////

    ////
    // Regular FFT: 1D
    ////

    inline xt::xarray<std::complex<long double> > fft (const xt::xarray<std::complex<long double> > &input) {
      return _fft_<std::complex<long double>, std::complex<long double>, 1, FFTW_FORWARD, true, false, false, fftwl_plan_dft_1d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > ifft (const xt::xarray<std::complex<long double> > &input) {
      return _ifft_<std::complex<long double>, std::complex<long double>, 1, FFTW_BACKWARD, true, false, false, fftwl_plan_dft_1d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    ////
    // Regular FFT: 2D
    ////

    inline xt::xarray<std::complex<long double> > fft2 (const xt::xarray<std::complex<long double> > &input) {
      return _fft_<std::complex<long double>, std::complex<long double>, 2, FFTW_FORWARD, true, false, false, fftwl_plan_dft_2d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > ifft2 (const xt::xarray<std::complex<long double> > &input) {
      return _ifft_<std::complex<long double>, std::complex<long double>, 2, FFTW_BACKWARD, true, false, false, fftwl_plan_dft_2d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    ////
    // Regular FFT: 3D
    ////

    inline xt::xarray<std::complex<long double> > fft3 (const xt::xarray<std::complex<long double> > &input) {
      return _fft_<std::complex<long double>, std::complex<long double>, 3, FFTW_FORWARD, true, false, false, fftwl_plan_dft_3d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > ifft3 (const xt::xarray<std::complex<long double> > &input) {
      return _ifft_<std::complex<long double>, std::complex<long double>, 3, FFTW_BACKWARD, true, false, false, fftwl_plan_dft_3d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    ////
    // Regular FFT: nD
    ////

    template <std::size_t dim>
    inline xt::xarray<std::complex<long double> > fftn (const xt::xarray<std::complex<long double> > &input) {
      return _fft_<std::complex<long double>, std::complex<long double>, dim, FFTW_FORWARD, false, false, false, fftwl_plan_dft, fftwl_execute, fftwl_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<std::complex<long double> > ifftn (const xt::xarray<std::complex<long double> > &input) {
      return _ifft_<std::complex<long double>, std::complex<long double>, dim, FFTW_BACKWARD, false, false, false, fftwl_plan_dft, fftwl_execute, fftwl_destroy_plan> (input);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Real FFT (real input)
    ///////////////////////////////////////////////////////////////////////////////

    ////
    // Real FFT: 1D
    ////

    inline xt::xarray<std::complex<long double> > rfft (const xt::xarray<long double> &input) {
      return _fft_<long double, std::complex<long double>, 1, 0, true, true, false, fftwl_plan_dft_r2c_1d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<long double> irfft (const xt::xarray<std::complex<long double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<long double>, long double, 1, 0, true, false, true, fftwl_plan_dft_c2r_1d, fftwl_execute, fftwl_destroy_plan> (input, odd_last_dim);
    }

    ////
    // Real FFT: 2D
    ////

    inline xt::xarray<std::complex<long double> > rfft2 (const xt::xarray<long double> &input) {
      return _fft_<long double, std::complex<long double>, 2, 0, true, true, false, fftwl_plan_dft_r2c_2d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<long double> irfft2 (const xt::xarray<std::complex<long double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<long double>, long double, 2, 0, true, false, true, fftwl_plan_dft_c2r_2d, fftwl_execute, fftwl_destroy_plan> (input, odd_last_dim);
    }

    ////
    // Real FFT: 3D
    ////

    inline xt::xarray<std::complex<long double> > rfft3 (const xt::xarray<long double> &input) {
      return _fft_<long double, std::complex<long double>, 3, 0, true, true, false, fftwl_plan_dft_r2c_3d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<long double> irfft3 (const xt::xarray<std::complex<long double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<long double>, long double, 3, 0, true, false, true, fftwl_plan_dft_c2r_3d, fftwl_execute, fftwl_destroy_plan> (input, odd_last_dim);
    }

    ////
    // Real FFT: nD
    ////

    template <std::size_t dim>
    inline xt::xarray<std::complex<long double> > rfftn (const xt::xarray<long double> &input) {
      return _fft_<long double, std::complex<long double>, dim, 0, false, true, false, fftwl_plan_dft_r2c, fftwl_execute, fftwl_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<long double> irfftn (const xt::xarray<std::complex<long double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<long double>, long double, dim, 0, false, false, true, fftwl_plan_dft_c2r, fftwl_execute, fftwl_destroy_plan> (input, odd_last_dim);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Hermitian FFT (real spectrum)
    ///////////////////////////////////////////////////////////////////////////////

    ////
    // Hermitian FFT: 1D
    ////

    inline xt::xarray<long double> hfft (const xt::xarray<std::complex<long double> > &input) {
      return _hfft_<std::complex<long double>, long double, 1, 0, true, false, true, fftwl_plan_dft_c2r_1d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > ihfft (const xt::xarray<long double> &input) {
      return _ihfft_<long double, std::complex<long double>, 1, 0, true, true, false, fftwl_plan_dft_r2c_1d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    ////
    // Hermitian FFT: 2D
    ////

    inline xt::xarray<long double> hfft2 (const xt::xarray<std::complex<long double> > &input) {
      return _hfft_<std::complex<long double>, long double, 2, 0, true, false, true, fftwl_plan_dft_c2r_2d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > ihfft2 (const xt::xarray<long double> &input) {
      return _ihfft_<long double, std::complex<long double>, 2, 0, true, true, false, fftwl_plan_dft_r2c_2d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    ////
    // Hermitian FFT: 3D
    ////

    inline xt::xarray<long double> hfft3 (const xt::xarray<std::complex<long double> > &input) {
      return _hfft_<std::complex<long double>, long double, 3, 0, true, false, true, fftwl_plan_dft_c2r_3d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > ihfft3 (const xt::xarray<long double> &input) {
      return _ihfft_<long double, std::complex<long double>, 3, 0, true, true, false, fftwl_plan_dft_r2c_3d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    ////
    // Hermitian FFT: nD
    ////

    template <std::size_t dim>
    inline xt::xarray<long double> hfftn (const xt::xarray<std::complex<long double> > &input) {
      return _hfft_<std::complex<long double>, long double, dim, 0, false, false, true, fftwl_plan_dft_c2r, fftwl_execute, fftwl_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<std::complex<long double> > ihfftn (const xt::xarray<long double> &input) {
      return _ihfft_<long double, std::complex<long double>, dim, 0, false, true, false, fftwl_plan_dft_r2c, fftwl_execute, fftwl_destroy_plan> (input);
    }

  }
}

#endif //XTENSOR_FFTW_BASIC_LONG_DOUBLE_HPP
