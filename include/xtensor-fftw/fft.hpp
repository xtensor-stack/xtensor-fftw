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
//    // TODO: replace T with some generic xarray<T> thing, so that I can get T = e.g. float out and use that for the Fourier-space output type (complex array of same precision)
//    template <typename T>
//    fourier_t<T> fft(T input) {
//
//    }
    xt::xarray<std::complex<float>> fft(const xt::xarray<float> &input) {
      xt::xarray<std::complex<float>> output(input.shape(), input.strides());
      fftwf_plan plan = fftwf_plan_dft_r2c_1d(static_cast<int>(input.size()), input, output, FFTW_ESTIMATE);
      fftwf_execute(plan);
      return output;
    }

    xt::xarray<float> ifft(xt::xarray<std::complex<float>> &input) {
      std::cout << "WARNING: the inverse c2r fftw transform by default destroys its input array, but in xt::fftw::ifft this has been disabled at the cost of some performance."
      xt::xarray<float> output(input.shape(), input.strides());
      fftwf_plan plan = fftwf_plan_dft_c2r_1d(static_cast<int>(input.size()), input, output, FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
      fftwf_execute(plan);
      return output;
    }
  }
}

#endif //XTENSOR_FFTW_FFT_HPP
