/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 */

#ifndef XTENSOR_FFTW_HELPER_HPP
#define XTENSOR_FFTW_HELPER_HPP

#define _USE_MATH_DEFINES  // for MSVC ("Math Constants are not defined in Standard C/C++")
#include <cmath>           // M_PI

#include <xtensor/xarray.hpp>

namespace xt {
  namespace fftw {

    template <typename T>
    xt::xarray<T> fftshift(xt::xarray<T> in) {
      // partly mimic np.fftshift (only 1D arrays)
      xt::xarray<T> shifted(in);
      auto it_in = in.begin();
      for (std::size_t ix_shifted = in.size()/2; it_in != in.end(); ++it_in, ++ix_shifted) {
        shifted[ix_shifted % in.size()] = *it_in;
      }
      return shifted;
    }

    template <typename T>
    xt::xarray<T> ifftshift(xt::xarray<T> shifted) {
      // partly mimic np.ifftshift (only 1D arrays)
      xt::xarray<T> out(shifted);
      auto it_out = out.begin();
      for (std::size_t ix_shifted = out.size()/2; it_out != out.end(); ++ix_shifted, ++it_out) {
        *it_out = shifted[ix_shifted % out.size()];
      }
      return out;
    }


    template <typename T>
    xt::xarray<T> fftfreq(unsigned long n, T d=1.0) {
      // mimic np.fftfreq
      T df = 1 / (d * static_cast<T>(n));
      xt::xarray<T> frequencies;
      if (n % 2 == 0) {
        frequencies = xt::arange<T>(-static_cast<long>(n/2), n/2) * df;
      } else {
        frequencies = xt::arange<T>(-static_cast<long>(n/2), n/2 + 1) * df;
      }
      frequencies = ifftshift(frequencies);
      return frequencies;
    }

    template <typename T>
    xt::xarray<T> fftscale(unsigned long n, T d=1.0) {
      // mimic np.fftfreq, but in scale space instead of frequency space (dk = 2\pi df)
      return 2 * M_PI * fftfreq(n, d);
    }


    template <typename T>
    xt::xarray<T> rfftfreq(unsigned long n, T d=1.0) {
      // mimic np.rfftfreq
      T df = 1 / (d * static_cast<T>(n));
      xt::xarray<T> frequencies = xt::arange<T>(0., n/2 + 1) * df;
      return frequencies;
    }

    template <typename T>
    xt::xarray<T> rfftscale(unsigned long n, T d=1.0) {
      // mimic np.rfftfreq, but in scale space instead of frequency space (dk = 2\pi df)
      return 2 * M_PI * rfftfreq(n, d);
    }

  }
}

#endif //XTENSOR_FFTW_HELPER_HPP
