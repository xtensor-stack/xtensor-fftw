/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * basic_double.hpp:
 * Contains the basic functions needed to do FFTs and inverse FFTs on double 
 * and complex<double> array. The behavior of these functions mimics that of
 * the numpy.fft module, see https://github.com/xtensor-stack/xtensor-fftw/issues/6.
 *
 */

#ifndef XTENSOR_FFTW_BASIC_DOUBLE_HPP
#define XTENSOR_FFTW_BASIC_DOUBLE_HPP

#include "common.hpp"

namespace xt {
  namespace fftw {

    template <> struct fftw_t<double> {
      using plan = fftw_plan;
      using complex = fftw_complex;
      constexpr static void (&execute)(plan) = fftw_execute;
      constexpr static void (&destroy_plan)(plan) = fftw_destroy_plan;
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Regular FFT (complex to complex)
    ///////////////////////////////////////////////////////////////////////////////

    ////
    // Regular FFT: 1D
    ////

    inline xt::xarray<std::complex<double> > fft (const xt::xarray<std::complex<double> > &input) {
      return _fft_<std::complex<double>, std::complex<double>, 1, FFTW_FORWARD, true, false, false, fftw_plan_dft_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > ifft (const xt::xarray<std::complex<double> > &input) {
      return _ifft_<std::complex<double>, std::complex<double>, 1, FFTW_BACKWARD, true, false, false, fftw_plan_dft_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    ////
    // Regular FFT: 2D
    ////

    inline xt::xarray<std::complex<double> > fft2 (const xt::xarray<std::complex<double> > &input) {
      return _fft_<std::complex<double>, std::complex<double>, 2, FFTW_FORWARD, true, false, false, fftw_plan_dft_2d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > ifft2 (const xt::xarray<std::complex<double> > &input) {
      return _ifft_<std::complex<double>, std::complex<double>, 2, FFTW_BACKWARD, true, false, false, fftw_plan_dft_2d, fftw_execute, fftw_destroy_plan> (input);
    }

    ////
    // Regular FFT: 3D
    ////

    inline xt::xarray<std::complex<double> > fft3 (const xt::xarray<std::complex<double> > &input) {
      return _fft_<std::complex<double>, std::complex<double>, 3, FFTW_FORWARD, true, false, false, fftw_plan_dft_3d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > ifft3 (const xt::xarray<std::complex<double> > &input) {
      return _ifft_<std::complex<double>, std::complex<double>, 3, FFTW_BACKWARD, true, false, false, fftw_plan_dft_3d, fftw_execute, fftw_destroy_plan> (input);
    }

    ////
    // Regular FFT: nD
    ////

    template <std::size_t dim>
    inline xt::xarray<std::complex<double> > fftn (const xt::xarray<std::complex<double> > &input) {
      return _fft_<std::complex<double>, std::complex<double>, dim, FFTW_FORWARD, false, false, false, fftw_plan_dft, fftw_execute, fftw_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<std::complex<double> > ifftn (const xt::xarray<std::complex<double> > &input) {
      return _ifft_<std::complex<double>, std::complex<double>, dim, FFTW_BACKWARD, false, false, false, fftw_plan_dft, fftw_execute, fftw_destroy_plan> (input);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Real FFT (real input)
    ///////////////////////////////////////////////////////////////////////////////

    ////
    // Real FFT: 1D
    ////

    inline xt::xarray<std::complex<double> > rfft (const xt::xarray<double> &input) {
      return _fft_<double, std::complex<double>, 1, 0, true, true, false, fftw_plan_dft_r2c_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<double> irfft (const xt::xarray<std::complex<double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<double>, double, 1, 0, true, false, true, fftw_plan_dft_c2r_1d, fftw_execute, fftw_destroy_plan> (input, odd_last_dim);
    }

    ////
    // Real FFT: 2D
    ////

    inline xt::xarray<std::complex<double> > rfft2 (const xt::xarray<double> &input) {
      return _fft_<double, std::complex<double>, 2, 0, true, true, false, fftw_plan_dft_r2c_2d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<double> irfft2 (const xt::xarray<std::complex<double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<double>, double, 2, 0, true, false, true, fftw_plan_dft_c2r_2d, fftw_execute, fftw_destroy_plan> (input, odd_last_dim);
    }

    ////
    // Real FFT: 3D
    ////

    inline xt::xarray<std::complex<double> > rfft3 (const xt::xarray<double> &input) {
      return _fft_<double, std::complex<double>, 3, 0, true, true, false, fftw_plan_dft_r2c_3d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<double> irfft3 (const xt::xarray<std::complex<double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<double>, double, 3, 0, true, false, true, fftw_plan_dft_c2r_3d, fftw_execute, fftw_destroy_plan> (input, odd_last_dim);
    }

    ////
    // Real FFT: nD
    ////

    template <std::size_t dim>
    inline xt::xarray<std::complex<double> > rfftn (const xt::xarray<double> &input) {
      return _fft_<double, std::complex<double>, dim, 0, false, true, false, fftw_plan_dft_r2c, fftw_execute, fftw_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<double> irfftn (const xt::xarray<std::complex<double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<double>, double, dim, 0, false, false, true, fftw_plan_dft_c2r, fftw_execute, fftw_destroy_plan> (input, odd_last_dim);
    }

    ///////////////////////////////////////////////////////////////////////////////
    // Hermitian FFT (real spectrum)
    ///////////////////////////////////////////////////////////////////////////////

    ////
    // Hermitian FFT: 1D
    ////

    inline xt::xarray<double> hfft (const xt::xarray<std::complex<double> > &input) {
      return _hfft_<std::complex<double>, double, 1, 0, true, false, true, fftw_plan_dft_c2r_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > ihfft (const xt::xarray<double> &input) {
      return _ihfft_<double, std::complex<double>, 1, 0, true, true, false, fftw_plan_dft_r2c_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    ////
    // Hermitian FFT: 2D
    ////

    inline xt::xarray<double> hfft2 (const xt::xarray<std::complex<double> > &input) {
      return _hfft_<std::complex<double>, double, 2, 0, true, false, true, fftw_plan_dft_c2r_2d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > ihfft2 (const xt::xarray<double> &input) {
      return _ihfft_<double, std::complex<double>, 2, 0, true, true, false, fftw_plan_dft_r2c_2d, fftw_execute, fftw_destroy_plan> (input);
    }

    ////
    // Hermitian FFT: 3D
    ////

    inline xt::xarray<double> hfft3 (const xt::xarray<std::complex<double> > &input) {
      return _hfft_<std::complex<double>, double, 3, 0, true, false, true, fftw_plan_dft_c2r_3d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > ihfft3 (const xt::xarray<double> &input) {
      return _ihfft_<double, std::complex<double>, 3, 0, true, true, false, fftw_plan_dft_r2c_3d, fftw_execute, fftw_destroy_plan> (input);
    }

    ////
    // Hermitian FFT: nD
    ////

    template <std::size_t dim>
    inline xt::xarray<double> hfftn (const xt::xarray<std::complex<double> > &input) {
      return _hfft_<std::complex<double>, double, dim, 0, false, false, true, fftw_plan_dft_c2r, fftw_execute, fftw_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<std::complex<double> > ihfftn (const xt::xarray<double> &input) {
      return _ihfft_<double, std::complex<double>, dim, 0, false, true, false, fftw_plan_dft_r2c, fftw_execute, fftw_destroy_plan> (input);
    }

  }
}

#endif //XTENSOR_FFTW_BASIC_DOUBLE_HPP
