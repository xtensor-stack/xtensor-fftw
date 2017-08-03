/*
 * xtensor-fftw
 *
 * Copyright 2017 Patrick Bos
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef XTENSOR_FFTW_FFT_HPP
#define XTENSOR_FFTW_FFT_HPP

#include <xtensor/xarray.hpp>
#include <complex>
#include <fftw3.h>

namespace xt {
  namespace fftw {
    // NOTE: when making more performant functions, keep in mind that many FFTW functions destroy input and/or output
    //       arrays! E.g. FFTW_MEASURE destroys both during measurement, c2r functions always destroy input by default
    //       (unless for 1D transforms if you pass FFTW_PRESERVE_INPUT,
    //       http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data),
    //       etc.
    //       http://www.fftw.org/fftw3_doc/Complex-One_002dDimensional-DFTs.html#Complex-One_002dDimensional-DFTs

    // We use template specializations to make sure we get the correct precision. The problem, for instance, with a
    // non-template f(const xt::xarray<float> &input) is that it will also compile when you pass a xt::xarray<double>.
    // Passing by const-reference in this sense behaves similarly to passing by value; it triggers the creation of a
    // temporary variable -- input in this case -- though in the case of a reference the data is not actually copied.
    // The delete makes sure that calls to non-implemented specializations don't compile. If this is left out, the
    // compilation will succeed, but the linker will fail, and this gives less informative error messages.
    // The inline keyword must be added to avoid multiple definition errors due to multiple compilations (e.g. when
    // including this header multiple times in a project, or when it is explicitly compiled itself and included too).
    template<typename real_t> inline xt::xarray< std::complex<real_t> > fft(const xt::xarray<real_t> &input) = delete;
    template<typename real_t> inline xt::xarray<real_t> ifft(const xt::xarray< std::complex<real_t> > &input) = delete;

    template<> inline xt::xarray<std::complex<float>> fft<float>(const xt::xarray<float> &input) {
      xt::xarray<std::complex<float>, layout_type::dynamic> output(input.shape(), input.strides());

      // this function will not modify input, see:
      // http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data
      // reinterpret_cast suggested by http://www.fftw.org/fftw3_doc/Complex-numbers.html
      fftwf_plan plan = fftwf_plan_dft_r2c_1d(static_cast<int>(input.size()),
                                              const_cast<float *>(input.raw_data()),
                                              reinterpret_cast<fftwf_complex*>(output.raw_data()),
                                              FFTW_ESTIMATE);

      fftwf_execute(plan);
      return output;
    }

    template<> inline xt::xarray<float> ifft<float>(const xt::xarray< std::complex<float> > &input) {
      std::cout << "WARNING: the inverse c2r fftw transform by default destroys its input array, but in xt::fftw::ifft this has been disabled at the cost of some performance." << std::endl;
      xt::xarray<float, layout_type::dynamic> output(input.shape(), input.strides());

      // this function will not modify input, see:
      // http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data
      // reinterpret_cast suggested by http://www.fftw.org/fftw3_doc/Complex-numbers.html
      fftwf_plan plan = fftwf_plan_dft_c2r_1d(static_cast<int>(input.size()),
                                              const_cast<fftwf_complex *>(reinterpret_cast<const fftwf_complex *>(input.raw_data())),
                                              output.raw_data(),
                                              FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

      fftwf_execute(plan);
      // we use the convention that the inverse fft divides by N, like numpy does
      return output / output.size();
    }

    template<> inline xt::xarray<std::complex<double>> fft<double>(const xt::xarray<double> &input) {
      xt::xarray<std::complex<double>, layout_type::dynamic> output(input.shape(), input.strides());

      // this function will not modify input, see:
      // http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data
      // reinterpret_cast suggested by http://www.fftw.org/fftw3_doc/Complex-numbers.html
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

      // this function will not modify input, see:
      // http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data
      // reinterpret_cast suggested by http://www.fftw.org/fftw3_doc/Complex-numbers.html
      fftw_plan plan = fftw_plan_dft_c2r_1d(static_cast<int>(input.size()),
                                            const_cast<fftw_complex *>(reinterpret_cast<const fftw_complex *>(input.raw_data())),
                                            output.raw_data(),
                                            FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

      fftw_execute(plan);
      // we use the convention that the inverse fft divides by N, like numpy does
      return output / output.size();
    }

    template<> inline xt::xarray<std::complex<long double>> fft<long double>(const xt::xarray<long double> &input) {
      xt::xarray<std::complex<long double>, layout_type::dynamic> output(input.shape(), input.strides());

      // this function will not modify input, see:
      // http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data
      // reinterpret_cast suggested by http://www.fftw.org/fftw3_doc/Complex-numbers.html
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

      // this function will not modify input, see:
      // http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data
      // reinterpret_cast suggested by http://www.fftw.org/fftw3_doc/Complex-numbers.html
      fftwl_plan plan = fftwl_plan_dft_c2r_1d(static_cast<int>(input.size()),
                                              const_cast<fftwl_complex *>(reinterpret_cast<const fftwl_complex *>(input.raw_data())),
                                              output.raw_data(),
                                              FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

      fftwl_execute(plan);
      // we use the convention that the inverse fft divides by N, like numpy does
      return output / output.size();
    }

  }
}

#endif //XTENSOR_FFTW_FFT_HPP
