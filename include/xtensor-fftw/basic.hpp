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


    ///////////////////////////////////////////////////////////////////////////////
    // General: templates defining the basic interaction logic with fftw. These
    //          will be specialized for all fft families, precisions and
    //          dimensionalities.
    ///////////////////////////////////////////////////////////////////////////////

    // aliases for the fftw precision-dependent types:
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
    // and subclass alias for when calling with a complex type:
    template <typename T> struct fftw_t< std::complex<T> > : public fftw_t<T> {};

    // convert std::complex to fftwX_complex with right precision X; non-complex floats stay themselves:
    template <typename regular_or_complex_t>
    using fftw_number_t = std::conditional_t<
        xtl::is_complex<regular_or_complex_t>::value,
        typename fftw_t< xtl::complex_value_type_t<regular_or_complex_t> >::complex,
        xtl::complex_value_type_t<regular_or_complex_t>
    >;

    // short-hand for precision for template arguments
    template <typename in_or_output_t>
    using prec_t = xtl::complex_value_type_t<in_or_output_t>;


    ////
    // General: xarray templates
    ////

    template<typename input_t, typename output_t, typename...>
    inline xt::xarray<output_t> _fft_ (const xt::xarray<input_t> &input) {
      static_assert(sizeof(prec_t<input_t>) == 0, "Only specializations of _fft_ can be used");
    }

    template<typename input_t, typename output_t, typename...>
    inline xt::xarray<output_t> _ifft_ (const xt::xarray<input_t> &input) {
      static_assert(sizeof(prec_t<input_t>) == 0, "Only specializations of _ifft_ can be used");
    }

    template <
        typename input_t, typename output_t,
        typename fftw_t<input_t>::plan (&fftw_plan_dft)(int, fftw_number_t<input_t> *, fftw_number_t<output_t> *, unsigned int),
        void (&fftw_execute)(typename fftw_t<input_t>::plan), void (&fftw_destroy_plan)(typename fftw_t<input_t>::plan),
        typename = std::enable_if_t<
            std::is_same< prec_t<input_t>, prec_t<output_t> >::value
            && std::is_floating_point< prec_t<input_t> >::value
        >
    >
    inline xt::xarray<output_t> _fft_(const xt::xarray<input_t> &input) {
      xt::xarray<output_t, layout_type::dynamic> output(input.shape(), input.strides());

      using fftw_input_t = fftw_number_t<input_t>;
      using fftw_output_t = fftw_number_t<output_t>;

      typename fftw_t<input_t>::plan plan = fftw_plan_dft(static_cast<int>(input.size()),
                                                          const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.raw_data())),
                                                          reinterpret_cast<fftw_output_t *>(output.raw_data()),
                                                          FFTW_ESTIMATE);

      fftw_execute(plan);
      fftw_destroy_plan(plan);
      return output;
    };

    template <
        typename input_t, typename output_t,
        typename fftw_t<input_t>::plan (&fftw_plan_dft)(int, fftw_number_t<input_t> *, fftw_number_t<output_t> *, unsigned int),
        void (&fftw_execute)(typename fftw_t<input_t>::plan), void (&fftw_destroy_plan)(typename fftw_t<input_t>::plan),
        typename = std::enable_if_t<
            std::is_same< prec_t<input_t>, prec_t<output_t> >::value
            && std::is_floating_point< prec_t<input_t> >::value
        >
    >
    inline xt::xarray<output_t> _ifft_(const xt::xarray<input_t> &input) {
      xt::xarray<output_t, layout_type::dynamic> output(input.shape(), input.strides());

      using fftw_input_t = fftw_number_t<input_t>;
      using fftw_output_t = fftw_number_t<output_t>;

      typename fftw_t<input_t>::plan plan = fftw_plan_dft(static_cast<int>(input.size()),
                                                          const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.raw_data())),
                                                          reinterpret_cast<fftw_output_t *>(output.raw_data()),
                                                          FFTW_ESTIMATE | FFTW_PRESERVE_INPUT);

      fftw_execute(plan);
      fftw_destroy_plan(plan);
      return output / output.size();
    };


    ////
    // General: xtensor templates
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


    ///////////////////////////////////////////////////////////////////////////////
    // Regular FFT (complex to complex)
    ///////////////////////////////////////////////////////////////////////////////

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

    inline xt::xarray<std::complex<float> > rfft (const xt::xarray<float> &input) {
      return _fft_<float, std::complex<float>, fftwf_plan_dft_r2c_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<float> irfft (const xt::xarray<std::complex<float> > &input) {
      return _ifft_<std::complex<float>, float, fftwf_plan_dft_c2r_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > rfft (const xt::xarray<double> &input) {
      return _fft_<double, std::complex<double>, fftw_plan_dft_r2c_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<double> irfft (const xt::xarray<std::complex<double> > &input) {
      return _ifft_<std::complex<double>, double, fftw_plan_dft_c2r_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > rfft (const xt::xarray<long double> &input) {
      return _fft_<long double, std::complex<long double>, fftwl_plan_dft_r2c_1d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<long double> irfft (const xt::xarray<std::complex<long double> > &input) {
      return _ifft_<std::complex<long double>, long double, fftwl_plan_dft_c2r_1d, fftwl_execute, fftwl_destroy_plan> (input);
    }


    ////
    // Real FFT: 2D
    ////

    inline xt::xarray<std::complex<float> > rfft2 (const xt::xarray<float> &input) {
      return _fft_<float, std::complex<float>, fftwf_plan_dft_r2c_2d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<float> irfft2 (const xt::xarray<std::complex<float> > &input) {
      return _ifft_<std::complex<float>, float, fftwf_plan_dft_c2r_2d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > rfft2 (const xt::xarray<double> &input) {
      return _fft_<double, std::complex<double>, fftw_plan_dft_r2c_2d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<double> irfft2 (const xt::xarray<std::complex<double> > &input) {
      return _ifft_<std::complex<double>, double, fftw_plan_dft_c2r_2d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > rfft2 (const xt::xarray<long double> &input) {
      return _fft_<long double, std::complex<long double>, fftwl_plan_dft_r2c_2d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<long double> irfft2 (const xt::xarray<std::complex<long double> > &input) {
      return _ifft_<std::complex<long double>, long double, fftwl_plan_dft_c2r_2d, fftwl_execute, fftwl_destroy_plan> (input);
    }


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
