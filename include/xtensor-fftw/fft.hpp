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

    xt::xarray<std::complex<float>> fft(const xt::xarray<float> &input) {
      xt::xarray<std::complex<float>, layout_type::dynamic> output(input.shape(), input.strides());
      fftwf_plan plan = fftwf_plan_dft_r2c_1d(static_cast<int>(input.size()),
                                              const_cast<float *>(input.raw_data()), // this function will not modify input, see http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data
                                              reinterpret_cast<fftwf_complex*>(output.raw_data()), // reinterpret_cast suggested by http://www.fftw.org/fftw3_doc/Complex-numbers.html
                                              FFTW_ESTIMATE);
      fftwf_execute(plan);
      return output;
    }

    xt::xarray<float> ifft(const xt::xarray<std::complex<float>> &input) {
      std::cout << "WARNING: the inverse c2r fftw transform by default destroys its input array, but in xt::fftw::ifft this has been disabled at the cost of some performance." << std::endl;
      xt::xarray<float, layout_type::dynamic> output(input.shape(), input.strides());
      fftwf_plan plan = fftwf_plan_dft_c2r_1d(static_cast<int>(input.size()),
                                              const_cast<fftwf_complex *>(reinterpret_cast<const fftwf_complex *>(input.raw_data())), // this function will not modify input, see http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data; reinterpret_cast suggested by http://www.fftw.org/fftw3_doc/Complex-numbers.html
                                              output.raw_data(),
                                              FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
      fftwf_execute(plan);
      return output;
    }
  }
}

#endif //XTENSOR_FFTW_FFT_HPP
