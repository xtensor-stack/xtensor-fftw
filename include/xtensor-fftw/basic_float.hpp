/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * basic_float.hpp:
 * Contains the basic functions needed to do FFTs and inverse FFTs on float
 * and complex<float> arrays. The behavior of these functions mimics that of 
 * the numpy.fft module, see https://github.com/xtensor-stack/xtensor-fftw/issues/6.
 *
 */

#ifndef XTENSOR_FFTW_BASIC_FLOAT_HPP
#define XTENSOR_FFTW_BASIC_FLOAT_HPP

#include "common.hpp"

namespace xt {
  namespace fftw {

    template <> struct fftw_t<float> {
      using plan = fftwf_plan;
      using complex = fftwf_complex;
      constexpr static void (&execute)(plan) = fftwf_execute;
      constexpr static void (&destroy_plan)(plan) = fftwf_destroy_plan;
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Regular FFT (complex to complex)
    ///////////////////////////////////////////////////////////////////////////////

    ////
    // Regular FFT: 1D
    ////

    inline xt::xarray<std::complex<float> > fft (const xt::xarray<std::complex<float> > &input) {
      return _fft_<std::complex<float>, std::complex<float>, 1, FFTW_FORWARD, true, false, false, fftwf_plan_dft_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<float> > ifft (const xt::xarray<std::complex<float> > &input) {
      return _ifft_<std::complex<float>, std::complex<float>, 1, FFTW_BACKWARD, true, false, false, fftwf_plan_dft_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    ////
    // Regular FFT: 2D
    ////

    inline xt::xarray<std::complex<float> > fft2 (const xt::xarray<std::complex<float> > &input) {
      return _fft_<std::complex<float>, std::complex<float>, 2, FFTW_FORWARD, true, false, false, fftwf_plan_dft_2d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<float> > ifft2 (const xt::xarray<std::complex<float> > &input) {
      return _ifft_<std::complex<float>, std::complex<float>, 2, FFTW_BACKWARD, true, false, false, fftwf_plan_dft_2d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    ////
    // Regular FFT: 3D
    ////

    inline xt::xarray<std::complex<float> > fft3 (const xt::xarray<std::complex<float> > &input) {
      return _fft_<std::complex<float>, std::complex<float>, 3, FFTW_FORWARD, true, false, false, fftwf_plan_dft_3d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<float> > ifft3 (const xt::xarray<std::complex<float> > &input) {
      return _ifft_<std::complex<float>, std::complex<float>, 3, FFTW_BACKWARD, true, false, false, fftwf_plan_dft_3d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    ////
    // Regular FFT: nD
    ////

    template <std::size_t dim>
    inline xt::xarray<std::complex<float> > fftn (const xt::xarray<std::complex<float> > &input) {
      return _fft_<std::complex<float>, std::complex<float>, dim, FFTW_FORWARD, false, false, false, fftwf_plan_dft, fftwf_execute, fftwf_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<std::complex<float> > ifftn (const xt::xarray<std::complex<float> > &input) {
      return _ifft_<std::complex<float>, std::complex<float>, dim, FFTW_BACKWARD, false, false, false, fftwf_plan_dft, fftwf_execute, fftwf_destroy_plan> (input);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Real FFT (real input)
    ///////////////////////////////////////////////////////////////////////////////

    ////
    // Real FFT: 1D
    ////

    inline xt::xarray<std::complex<float> > rfft (const xt::xarray<float> &input) {
      return _fft_<float, std::complex<float>, 1, 0, true, true, false, fftwf_plan_dft_r2c_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<float> irfft (const xt::xarray<std::complex<float> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<float>, float, 1, 0, true, false, true, fftwf_plan_dft_c2r_1d, fftwf_execute, fftwf_destroy_plan> (input, odd_last_dim);
    }

    ////
    // Real FFT: 2D
    ////

    inline xt::xarray<std::complex<float> > rfft2 (const xt::xarray<float> &input) {
      return _fft_<float, std::complex<float>, 2, 0, true, true, false, fftwf_plan_dft_r2c_2d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<float> irfft2 (const xt::xarray<std::complex<float> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<float>, float, 2, 0, true, false, true, fftwf_plan_dft_c2r_2d, fftwf_execute, fftwf_destroy_plan> (input, odd_last_dim);
    }

    ////
    // Real FFT: 3D
    ////

    inline xt::xarray<std::complex<float> > rfft3 (const xt::xarray<float> &input) {
      return _fft_<float, std::complex<float>, 3, 0, true, true, false, fftwf_plan_dft_r2c_3d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<float> irfft3 (const xt::xarray<std::complex<float> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<float>, float, 3, 0, true, false, true, fftwf_plan_dft_c2r_3d, fftwf_execute, fftwf_destroy_plan> (input, odd_last_dim);
    }

    ////
    // Real FFT: nD
    ////

    template <std::size_t dim>
    inline xt::xarray<std::complex<float> > rfftn (const xt::xarray<float> &input) {
      return _fft_<float, std::complex<float>, dim, 0, false, true, false, fftwf_plan_dft_r2c, fftwf_execute, fftwf_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<float> irfftn (const xt::xarray<std::complex<float> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<float>, float, dim, 0, false, false, true, fftwf_plan_dft_c2r, fftwf_execute, fftwf_destroy_plan> (input, odd_last_dim);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Hermitian FFT (real spectrum)
    ///////////////////////////////////////////////////////////////////////////////

    ////
    // Hermitian FFT: 1D
    ////

    inline xt::xarray<float> hfft (const xt::xarray<std::complex<float> > &input) {
      return _hfft_<std::complex<float>, float, 1, 0, true, false, true, fftwf_plan_dft_c2r_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<float> > ihfft (const xt::xarray<float> &input) {
      return _ihfft_<float, std::complex<float>, 1, 0, true, true, false, fftwf_plan_dft_r2c_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    ////
    // Hermitian FFT: 2D
    ////

    inline xt::xarray<float> hfft2 (const xt::xarray<std::complex<float> > &input) {
      return _hfft_<std::complex<float>, float, 2, 0, true, false, true, fftwf_plan_dft_c2r_2d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<float> > ihfft2 (const xt::xarray<float> &input) {
      return _ihfft_<float, std::complex<float>, 2, 0, true, true, false, fftwf_plan_dft_r2c_2d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    ////
    // Hermitian FFT: 3D
    ////

    inline xt::xarray<float> hfft3 (const xt::xarray<std::complex<float> > &input) {
      return _hfft_<std::complex<float>, float, 3, 0, true, false, true, fftwf_plan_dft_c2r_3d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<float> > ihfft3 (const xt::xarray<float> &input) {
      return _ihfft_<float, std::complex<float>, 3, 0, true, true, false, fftwf_plan_dft_r2c_3d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    ////
    // Hermitian FFT: nD
    ////

    template <std::size_t dim>
    inline xt::xarray<float> hfftn (const xt::xarray<std::complex<float> > &input) {
      return _hfft_<std::complex<float>, float, dim, 0, false, false, true, fftwf_plan_dft_c2r, fftwf_execute, fftwf_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<std::complex<float> > ihfftn (const xt::xarray<float> &input) {
      return _ihfft_<float, std::complex<float>, dim, 0, false, true, false, fftwf_plan_dft_r2c, fftwf_execute, fftwf_destroy_plan> (input);
    }

  }
}

#endif //XTENSOR_FFTW_BASIC_FLOAT_HPP
