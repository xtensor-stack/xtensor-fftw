/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * basic.hpp:
 * Contains the basic functions needed to do FFTs and inverse FFTs on real and
 * complex arrays. The behavior of these functions mimics that of the numpy.fft
 * module, see https://github.com/egpbos/xtensor-fftw/issues/6.
 *
 */

#ifndef XTENSOR_FFTW_BASIC_HPP
#define XTENSOR_FFTW_BASIC_HPP

#include <xtensor/xarray.hpp>
#include <xtl/xcomplex.hpp>
#include <complex>
#include <type_traits>
#include <fftw3.h>

namespace xt {
  namespace fftw {
    // The implementations must be inline to avoid multiple definition errors due to multiple compilations (e.g. when
    // including this header multiple times in a project, or when it is explicitly compiled itself and included too).

    // Note: multidimensional complex-to-real transforms by default destroy the input data! See:
    // http://www.fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html#One_002dDimensional-DFTs-of-Real-Data

    // reinterpret_casts below suggested by http://www.fftw.org/fftw3_doc/Complex-numbers.html

    // We use the convention that the inverse fft divides by N, like numpy does.

    ////
    // aliases for the fftw precision-dependent types
    ////

    template <typename T> struct fftw_t {
      static_assert(sizeof(T) == 0, "Only specializations of fftw_t can be used");
    };
    template <> struct fftw_t<float> {
      using plan = fftwf_plan;
      using complex = fftwf_complex;
    };
    template <> struct fftw_t<double> {
      using plan = fftw_plan;
      using complex = fftw_complex;
    };
    template <> struct fftw_t<long double> {
      using plan = fftwl_plan;
      using complex = fftwl_complex;
    };

    ///////////////////////////////////////////////////////////////////////////////
    // Regular FFT (complex to complex)
    ///////////////////////////////////////////////////////////////////////////////

    ////
    // Regular FFT: xarray templates
    ////

    template<typename precision_t, typename input_t, typename output_t, typename...>
    xt::xarray<output_t> _fft_ (const xt::xarray<input_t> &input) {
      static_assert(sizeof(precision_t) == 0, "Only specializations of _fft_ can be used");
    }

    template<typename precision_t, typename input_t, typename output_t, typename...>
    xt::xarray<output_t> _ifft_ (const xt::xarray<input_t> &input) {
      static_assert(sizeof(precision_t) == 0, "Only specializations of _ifft_ can be used");
    }

    template<typename precision_t, typename input_t, typename output_t,
        typename fftw_t<precision_t>::plan (&fftw_plan_dft)(
            int,
            std::conditional_t<xtl::is_complex<input_t>::value, typename fftw_t<precision_t>::complex*, precision_t*>,
            std::conditional_t<xtl::is_complex<output_t>::value, typename fftw_t<precision_t>::complex*, precision_t*>,
            unsigned int),
        void (&fftw_execute)(typename fftw_t<precision_t>::plan), void (&fftw_destroy_plan)(typename fftw_t<precision_t>::plan)>
    xt::xarray<output_t> _fft_(const xt::xarray<input_t> &input) {
      xt::xarray<output_t, layout_type::dynamic> output(input.shape(), input.strides());

      using fftw_input_t = std::conditional_t<xtl::is_complex<input_t>::value, typename fftw_t<precision_t>::complex, precision_t>;
      using fftw_output_t = std::conditional_t<xtl::is_complex<output_t>::value, typename fftw_t<precision_t>::complex, precision_t>;

      typename fftw_t<precision_t>::plan plan = fftw_plan_dft(static_cast<int>(input.size()),
                                                     const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.raw_data())),
                                                     reinterpret_cast<fftw_output_t *>(output.raw_data()),
                                                     FFTW_ESTIMATE);

      fftw_execute(plan);
      fftw_destroy_plan(plan);
      return output;
    };

    template<typename precision_t, typename input_t, typename output_t,
        typename fftw_t<precision_t>::plan (&fftw_plan_dft)(
            int,
            std::conditional_t<xtl::is_complex<input_t>::value, typename fftw_t<precision_t>::complex*, precision_t*>,
            std::conditional_t<xtl::is_complex<output_t>::value, typename fftw_t<precision_t>::complex*, precision_t*>,
            unsigned int),
        void (&fftw_execute)(typename fftw_t<precision_t>::plan), void (&fftw_destroy_plan)(typename fftw_t<precision_t>::plan)>
    xt::xarray<output_t> _ifft_(const xt::xarray<input_t> &input) {
      xt::xarray<output_t, layout_type::dynamic> output(input.shape(), input.strides());

      using fftw_input_t = std::conditional_t<xtl::is_complex<input_t>::value, typename fftw_t<precision_t>::complex, precision_t>;
      using fftw_output_t = std::conditional_t<xtl::is_complex<output_t>::value, typename fftw_t<precision_t>::complex, precision_t>;

      typename fftw_t<precision_t>::plan plan = fftw_plan_dft(static_cast<int>(input.size()),
                                                     const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.raw_data())),
                                                     reinterpret_cast<fftw_output_t *>(output.raw_data()),
                                                     FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

      fftw_execute(plan);
      fftw_destroy_plan(plan);
      return output / output.size();
    };

//    xt::xarray<std::complex<float>> _fft_<float, float, std::complex<float>, fftwf_plan (&fftw_plan_dft_r2c_1d)(int, float*, fftwf_complex*, unsigned int), void (&fftwf_execute)(fftwf_plan), void (&fftwf_destroy_plan)(fftwf_plan) > (const xt::xarray<float> &input);
//    template<>
//    xt::xarray<std::complex<float> > _fft_<float, float, std::complex<float>, fftwf_plan_dft_r2c_1d, fftwf_execute, fftwf_destroy_plan> (const xt::xarray<float> &input);
//    template<>
//    xt::xarray<float> _ifft_<float, std::complex<float>, float, fftwf_plan_dft_c2r_1d, fftwf_execute, fftwf_destroy_plan> (const xt::xarray<std::complex<float> > &input);

    inline xt::xarray<std::complex<float> > RFFT (const xt::xarray<float> &input) {
      return _fft_<float, float, std::complex<float>, fftwf_plan_dft_r2c_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<float> IRFFT (const xt::xarray<std::complex<float> > &input) {
      return _ifft_<float, std::complex<float>, float, fftwf_plan_dft_c2r_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > RFFT (const xt::xarray<double> &input) {
      return _fft_<double, double, std::complex<double>, fftw_plan_dft_r2c_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<double> IRFFT (const xt::xarray<std::complex<double> > &input) {
      return _ifft_<double, std::complex<double>, double, fftw_plan_dft_c2r_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > RFFT (const xt::xarray<long double> &input) {
      return _fft_<long double, long double, std::complex<long double>, fftwl_plan_dft_r2c_1d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<long double> IRFFT (const xt::xarray<std::complex<long double> > &input) {
      return _ifft_<long double, std::complex<long double>, long double, fftwl_plan_dft_c2r_1d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    ////
    // Regular FFT: xtensor templates
    ////

//    template<typename real_t, std::size_t dim, typename fftw_plan_t>
//    xt::xtensor< std::complex<real_t>, dim > _fft_(const xt::xtensor<real_t, dim> &input) {
//      static_assert(sizeof(real_t) == 0, "Only specializations of fft can be used");
//
//      xt::xtensor<std::complex<real_t>, dim> output(input.shape(), input.strides());
//
//      fftw_plan_t plan = fftwXXXXX_plan_dft_r2c_1d(static_cast<int>(input.size()),
//                                              const_cast<real_t *>(input.raw_data()),
//                                              reinterpret_cast<fftwXXXXXXX_complex*>(output.raw_data()),
//                                              FFTW_ESTIMATE);
//
//      fftwXXXXX_execute(plan);
//      fftwXXXXX_destroy_plan(plan);
//      return output;
//    };
//
//    template<typename real_t, std::size_t dim, typename fftw_plan_t>
//    xt::xtensor<real_t, dim> _ifft_(const xt::xtensor< std::complex<real_t>, dim > &input) {
//      static_assert(sizeof(real_t) == 0, "Only specializations of ifft can be used");
//
//      std::cout << "WARNING: the inverse c2r fftw transform by default destroys its input array, but in xt::fftw::ifft this has been disabled at the cost of some performance." << std::endl;
//      xt::xtensor<real_t, dim> output(input.shape(), input.strides());
//
//      fftw_plan_t plan = fftwXXXXX_plan_dft_c2r_1d(static_cast<int>(input.size()),
//                                                      const_cast<fftwXXXXX_complex *>(reinterpret_cast<const fftwXXXXX_complex *>(input.raw_data())),
//                                                      output.raw_data(),
//                                                      FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);
//
//      fftwXXXXX_execute(plan);
//      fftwXXXXX_destroy_plan(plan);
//      return output / output.size();
//    };
    ////
    // Regular FFT: 1D
    ////


    ////
    // Regular FFT: 2D
    ////


    ////
    // Regular FFT: 3D
    ////


    ////
    // Regular FFT: nD
    ////


    ///////////////////////////////////////////////////////////////////////////////
    // Real FFT (real input)
    ///////////////////////////////////////////////////////////////////////////////

    ////
    // Real FFT: 1D
    ////

    template<typename real_t>
    xt::xarray< std::complex<real_t> > rfft(const xt::xarray<real_t> &input) {
      static_assert(sizeof(real_t) == 0, "Only specializations of fft can be used");
    };
    template<typename real_t>
    xt::xarray<real_t> irfft(const xt::xarray< std::complex<real_t> > &input) {
      static_assert(sizeof(real_t) == 0, "Only specializations of ifft can be used");
    };

    template<>
    inline xt::xarray< std::complex<float> > rfft<float>(const xt::xarray<float> &input) {
      xt::xarray<std::complex<float>, layout_type::dynamic> output(input.shape(), input.strides());

      fftwf_plan plan = fftwf_plan_dft_r2c_1d(static_cast<int>(input.size()),
                                              const_cast<float *>(input.raw_data()),
                                              reinterpret_cast<fftwf_complex*>(output.raw_data()),
                                              FFTW_ESTIMATE);

      fftwf_execute(plan);
      fftwf_destroy_plan(plan);
      return output;
    }

    template<>
    inline xt::xarray<float> irfft<float>(const xt::xarray< std::complex<float> > &input) {
      std::cout << "WARNING: the inverse c2r fftw transform by default destroys its input array, but in xt::fftw::ifft this has been disabled at the cost of some performance." << std::endl;
      xt::xarray<float, layout_type::dynamic> output(input.shape(), input.strides());

      fftwf_plan plan = fftwf_plan_dft_c2r_1d(static_cast<int>(input.size()),
                                              const_cast<fftwf_complex *>(reinterpret_cast<const fftwf_complex *>(input.raw_data())),
                                              output.raw_data(),
                                              FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

      fftwf_execute(plan);
      fftwf_destroy_plan(plan);
      return output / output.size();
    }

    template<> inline xt::xarray< std::complex<double> > rfft<double>(const xt::xarray<double> &input) {
      xt::xarray<std::complex<double>, layout_type::dynamic> output(input.shape(), input.strides());

      fftw_plan plan = fftw_plan_dft_r2c_1d(static_cast<int>(input.size()),
                                            const_cast<double *>(input.raw_data()),
                                            reinterpret_cast<fftw_complex*>(output.raw_data()),
                                            FFTW_ESTIMATE);

      fftw_execute(plan);
      fftw_destroy_plan(plan);
      return output;
    }

    template<> inline xt::xarray<double> irfft<double>(const xt::xarray< std::complex<double> > &input) {
      std::cout << "WARNING: the inverse c2r fftw transform by default destroys its input array, but in xt::fftw::ifft this has been disabled at the cost of some performance." << std::endl;
      xt::xarray<double, layout_type::dynamic> output(input.shape(), input.strides());

      fftw_plan plan = fftw_plan_dft_c2r_1d(static_cast<int>(input.size()),
                                            const_cast<fftw_complex *>(reinterpret_cast<const fftw_complex *>(input.raw_data())),
                                            output.raw_data(),
                                            FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

      fftw_execute(plan);
      fftw_destroy_plan(plan);
      return output / output.size();
    }

    template<> inline xt::xarray< std::complex<long double> > rfft<long double>(const xt::xarray<long double> &input) {
      xt::xarray<std::complex<long double>, layout_type::dynamic> output(input.shape(), input.strides());

      fftwl_plan plan = fftwl_plan_dft_r2c_1d(static_cast<int>(input.size()),
                                              const_cast<long double *>(input.raw_data()),
                                              reinterpret_cast<fftwl_complex*>(output.raw_data()),
                                              FFTW_ESTIMATE);

      fftwl_execute(plan);
      fftwl_destroy_plan(plan);
      return output;
    }

    template<> inline xt::xarray<long double> irfft<long double>(const xt::xarray< std::complex<long double> > &input) {
      std::cout << "WARNING: the inverse c2r fftw transform by default destroys its input array, but in xt::fftw::ifft this has been disabled at the cost of some performance." << std::endl;
      xt::xarray<long double, layout_type::dynamic> output(input.shape(), input.strides());

      fftwl_plan plan = fftwl_plan_dft_c2r_1d(static_cast<int>(input.size()),
                                              const_cast<fftwl_complex *>(reinterpret_cast<const fftwl_complex *>(input.raw_data())),
                                              output.raw_data(),
                                              FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

      fftwl_execute(plan);
      fftwl_destroy_plan(plan);
      return output / output.size();
    }


    ////
    // Real FFT: 2D
    ////


    ////
    // Real FFT: 3D
    ////


    ////
    // Real FFT: nD
    ////


    ///////////////////////////////////////////////////////////////////////////////
    // Hermitian FFT (real spectrum)
    ///////////////////////////////////////////////////////////////////////////////

    ////
    // Hermitian FFT: 1D
    ////


    ////
    // Hermitian FFT: 2D
    ////


    ////
    // Hermitian FFT: 3D
    ////


    ////
    // Hermitian FFT: nD
    ////


  }
}

#endif //XTENSOR_FFTW_BASIC_HPP
