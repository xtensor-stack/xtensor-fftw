/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * fft.hpp:
 * Contains the basic functions needed to do FFTs and inverse FFTs on real and
 * complex arrays. The output of inverse Fourier transformations is always a
 * complex array. The behavior of these functions mimics that of the numpy.fft
 * module, see https://github.com/egpbos/xtensor-fftw/issues/6.
 *
 */

#ifndef XTENSOR_FFTW_FFT_HPP
#define XTENSOR_FFTW_FFT_HPP

#include <xtensor/xarray.hpp>
#include <complex>
#include <fftw3.h>

namespace xt {
  namespace fftw {
    // The implementations must be inline to avoid multiple definition errors due to multiple compilations (e.g. when
    // including this header multiple times in a project, or when it is explicitly compiled itself and included too).

    // Note: multidimensional complex-to-real transforms by default destroy the input data! See:
    // http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data

    // reinterpret_casts below suggested by http://www.fftw.org/fftw3_doc/Complex-numbers.html

    // We use the convention that the inverse fft divides by N, like numpy does.

    //////////////////
    /////// 1D ///////
    //////////////////

    template<typename real_t>
    xt::xarray< std::complex<real_t> > fft(const xt::xarray<real_t> &input) {
      static_assert(sizeof(real_t) == 0, "Only specializations of fft can be used");
    };
    template<typename real_t>
    xt::xarray<real_t> ifft(const xt::xarray< std::complex<real_t> > &input) {
      static_assert(sizeof(real_t) == 0, "Only specializations of ifft can be used");
    };

    template<>
    inline xt::xarray< std::complex<float> > fft<float>(const xt::xarray<float> &input) {
      xt::xarray<std::complex<float>, layout_type::dynamic> output(input.shape(), input.strides());

      fftwf_plan plan = fftwf_plan_dft_r2c_1d(static_cast<int>(input.size()),
                                              const_cast<float *>(input.raw_data()),
                                              reinterpret_cast<fftwf_complex*>(output.raw_data()),
                                              FFTW_ESTIMATE);

      fftwf_execute(plan);
      return output;
    }

    template<>
    inline xt::xarray<float> ifft<float>(const xt::xarray< std::complex<float> > &input) {
      std::cout << "WARNING: the inverse c2r fftw transform by default destroys its input array, but in xt::fftw::ifft this has been disabled at the cost of some performance." << std::endl;
      xt::xarray<float, layout_type::dynamic> output(input.shape(), input.strides());

      fftwf_plan plan = fftwf_plan_dft_c2r_1d(static_cast<int>(input.size()),
                                              const_cast<fftwf_complex *>(reinterpret_cast<const fftwf_complex *>(input.raw_data())),
                                              output.raw_data(),
                                              FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

      fftwf_execute(plan);
      return output / output.size();
    }

    template<> inline xt::xarray< std::complex<double> > fft<double>(const xt::xarray<double> &input) {
      xt::xarray<std::complex<double>, layout_type::dynamic> output(input.shape(), input.strides());

      fftw_plan plan = fftw_plan_dft_r2c_1d(static_cast<int>(input.size()),
                                            const_cast<double *>(input.raw_data()),
                                            reinterpret_cast<fftw_complex*>(output.raw_data()),
                                            FFTW_ESTIMATE);

      fftw_execute(plan);
      return output;
    }

    template<> inline xt::xarray<double> ifft<double>(const xt::xarray< std::complex<double> > &input) {
      std::cout << "WARNING: the inverse c2r fftw transform by default destroys its input array, but in xt::fftw::ifft this has been disabled at the cost of some performance." << std::endl;
      xt::xarray<double, layout_type::dynamic> output(input.shape(), input.strides());

      fftw_plan plan = fftw_plan_dft_c2r_1d(static_cast<int>(input.size()),
                                            const_cast<fftw_complex *>(reinterpret_cast<const fftw_complex *>(input.raw_data())),
                                            output.raw_data(),
                                            FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

      fftw_execute(plan);
      return output / output.size();
    }

    template<> inline xt::xarray< std::complex<long double> > fft<long double>(const xt::xarray<long double> &input) {
      xt::xarray<std::complex<long double>, layout_type::dynamic> output(input.shape(), input.strides());

      fftwl_plan plan = fftwl_plan_dft_r2c_1d(static_cast<int>(input.size()),
                                              const_cast<long double *>(input.raw_data()),
                                              reinterpret_cast<fftwl_complex*>(output.raw_data()),
                                              FFTW_ESTIMATE);

      fftwl_execute(plan);
      return output;
    }

    template<> inline xt::xarray<long double> ifft<long double>(const xt::xarray< std::complex<long double> > &input) {
      std::cout << "WARNING: the inverse c2r fftw transform by default destroys its input array, but in xt::fftw::ifft this has been disabled at the cost of some performance." << std::endl;
      xt::xarray<long double, layout_type::dynamic> output(input.shape(), input.strides());

      fftwl_plan plan = fftwl_plan_dft_c2r_1d(static_cast<int>(input.size()),
                                              const_cast<fftwl_complex *>(reinterpret_cast<const fftwl_complex *>(input.raw_data())),
                                              output.raw_data(),
                                              FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

      fftwl_execute(plan);
      return output / output.size();
    }

    //////////////////
    /////// 2D ///////
    //////////////////

    //////////////////
    /////// 3D ///////
    //////////////////

    //////////////////////////
    /////// nD xtensor ///////
    //////////////////////////

    //////////////////////
    /////// xarray ///////
    //////////////////////

  }
}

#endif //XTENSOR_FFTW_FFT_HPP
