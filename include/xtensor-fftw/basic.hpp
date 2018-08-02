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
#include "xtensor/xcomplex.hpp"
#include "xtensor/xeval.hpp"
#include <xtl/xcomplex.hpp>
#include <complex>
#include <tuple>
#include <type_traits>
#include <exception>

// for product accumulate:
#include <numeric>
#include <functional>

#include <fftw3.h>

#ifdef __CLING__
  #pragma cling load("fftw3")
#endif

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
      constexpr static void (&execute)(plan) = fftwf_execute;
      constexpr static void (&destroy_plan)(plan) = fftwf_destroy_plan;
    };
    template <> struct fftw_t<double> {
      using plan = fftw_plan;
      using complex = fftw_complex;
      constexpr static void (&execute)(plan) = fftw_execute;
      constexpr static void (&destroy_plan)(plan) = fftw_destroy_plan;
    };
    template <> struct fftw_t<long double> {
      using plan = fftwl_plan;
      using complex = fftwl_complex;
      constexpr static void (&execute)(plan) = fftwl_execute;
      constexpr static void (&destroy_plan)(plan) = fftwl_destroy_plan;
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

    // dimension-dependent function signatures of fftw planning functions
    template <typename input_t, typename output_t, std::size_t dim, int fftw_direction, bool fftw_123dim>
    struct fftw_plan_dft_signature {};

    template <typename input_t, typename output_t, std::size_t dim>
    struct fftw_plan_dft_signature<input_t, output_t, dim, 0, false> {
      using type = typename fftw_t<input_t>::plan (&)(int rank, const int *n, fftw_number_t<input_t> *, fftw_number_t<output_t> *, unsigned int);
    };
    template <typename input_t, typename output_t>
    struct fftw_plan_dft_signature<input_t, output_t, 1, 0, true> {
      using type = typename fftw_t<input_t>::plan (&)(int n1, fftw_number_t<input_t> *, fftw_number_t<output_t> *, unsigned int);
    };
    template <typename input_t, typename output_t>
    struct fftw_plan_dft_signature<input_t, output_t, 2, 0, true> {
      using type = typename fftw_t<input_t>::plan (&)(int n1, int n2, fftw_number_t<input_t> *, fftw_number_t<output_t> *, unsigned int);
    };
    template <typename input_t, typename output_t>
    struct fftw_plan_dft_signature<input_t, output_t, 3, 0, true> {
      using type = typename fftw_t<input_t>::plan (&)(int n1, int n2, int n3, fftw_number_t<input_t> *, fftw_number_t<output_t> *, unsigned int);
    };

    template <typename input_t, typename output_t, std::size_t dim, int fftw_direction>
    struct fftw_plan_dft_signature<input_t, output_t, dim, fftw_direction, false> {
      using type = typename fftw_t<input_t>::plan (&)(int rank, const int *n, fftw_number_t<input_t> *, fftw_number_t<output_t> *, int, unsigned int);
    };
    template <typename input_t, typename output_t, int fftw_direction>
    struct fftw_plan_dft_signature<input_t, output_t, 1, fftw_direction, true> {
      using type = typename fftw_t<input_t>::plan (&)(int n1, fftw_number_t<input_t> *, fftw_number_t<output_t> *, int, unsigned int);
    };
    template <typename input_t, typename output_t, int fftw_direction>
    struct fftw_plan_dft_signature<input_t, output_t, 2, fftw_direction, true> {
      using type = typename fftw_t<input_t>::plan (&)(int n1, int n2, fftw_number_t<input_t> *, fftw_number_t<output_t> *, int, unsigned int);
    };
    template <typename input_t, typename output_t, int fftw_direction>
    struct fftw_plan_dft_signature<input_t, output_t, 3, fftw_direction, true> {
      using type = typename fftw_t<input_t>::plan (&)(int n1, int n2, int n3, fftw_number_t<input_t> *, fftw_number_t<output_t> *, int, unsigned int);
    };


    // all_true, from https://stackoverflow.com/a/28253503/1199693
    template <bool...> struct bool_pack;

    template <bool... v>
    using all_true = std::is_same< bool_pack<true, v...>, bool_pack<v..., true> >;

    // conditionals for correct combinations of dimensionality parameters
    namespace dimensional {
      template <std::size_t dim, bool fftw_123dim>
      struct is_1 : public std::false_type {};
      template <>
      struct is_1<1, true> : public std::true_type {};

      template <std::size_t dim, bool fftw_123dim>
      struct is_2 : public std::false_type {};
      template <>
      struct is_2<2, true> : public std::true_type {};

      template <std::size_t dim, bool fftw_123dim>
      struct is_3 : public std::false_type {};
      template <>
      struct is_3<3, true> : public std::true_type {};

      template <std::size_t dim, bool fftw_123dim>
      struct is_123 : public std::conditional_t<
          is_1<dim, fftw_123dim>::value || is_2<dim, fftw_123dim>::value || is_3<dim, fftw_123dim>::value,
          std::true_type,
          std::false_type
      > {};

      template <std::size_t dim, bool fftw_123dim>
      struct is_n : public std::false_type {};
      template <std::size_t dim>
      struct is_n<dim, false> : public std::true_type {};
    }


    // input vs output shape conversion
    template <typename output_t>
    inline typename xt::xarray<output_t, xt::layout_type::row_major>::shape_type output_shape_from_input(xt::xarray<output_t, xt::layout_type::row_major> input, bool half_plus_one_out, bool half_plus_one_in, bool odd_last_dim = false) {
      auto output_shape = input.shape();
      if (half_plus_one_out) {        // r2c
        auto n = output_shape.size();
        output_shape[n-1] = output_shape[n-1]/2 + 1;
      } else if (half_plus_one_in) {  // c2r
        auto n = output_shape.size();
        if (!odd_last_dim) {
          output_shape[n - 1] = (output_shape[n - 1] - 1) * 2;
        } else {
          output_shape[n - 1] = (output_shape[n - 1] - 1) * 2 + 1;
        }
      }
      return output_shape;
    }

    // output to DFT-dimensions conversion
    template <typename output_t>
    inline typename xt::xarray<output_t, xt::layout_type::row_major>::shape_type dft_dimensions_from_output(xt::xarray<output_t, xt::layout_type::row_major> output, bool half_plus_one_out, bool odd_last_dim = false) {
      auto dft_dimensions = output.shape();

      if (half_plus_one_out) {        // r2c
        auto n = dft_dimensions.size();
        if (!odd_last_dim) {
          dft_dimensions[n - 1] = (dft_dimensions[n - 1] - 1) * 2;
        } else {
          dft_dimensions[n - 1] = (dft_dimensions[n - 1] - 1) * 2 + 1;
        }
      }

      return dft_dimensions;
    }


    // Callers for fftw_plan_dft, since they have different call signatures and the
    // way shape information is extracted from xtensor differs for different dimensionalities.

    // REGULAR FFT N-dim
    template <std::size_t dim, int fftw_direction, bool fftw_123dim, typename input_t, typename output_t, typename fftw_plan_dft_signature<input_t, output_t, dim, fftw_direction, fftw_123dim>::type fftw_plan_dft, bool half_plus_one_out, bool half_plus_one_in>
    inline auto fftw_plan_dft_caller(const xt::xarray<input_t, layout_type::row_major> &input, xt::xarray<output_t, layout_type::row_major> &output, unsigned int flags, bool /*odd_last_dim*/ = false)
    -> std::enable_if_t<dimensional::is_n<dim, fftw_123dim>::value && (fftw_direction != 0), typename fftw_t<input_t>::plan> {
      using fftw_input_t = fftw_number_t<input_t>;
      using fftw_output_t = fftw_number_t<output_t>;

      auto dft_dimensions_unsigned = dft_dimensions_from_output(output, half_plus_one_out);
      std::vector<int> dft_dimensions;
      dft_dimensions.reserve(dft_dimensions_unsigned.size());
      std::transform(dft_dimensions_unsigned.begin(), dft_dimensions_unsigned.end(), std::back_inserter(dft_dimensions), [&](std::size_t d) { return static_cast<int>(d); });

      return fftw_plan_dft(static_cast<int>(dim), dft_dimensions.data(),
                           const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.data())),
                           reinterpret_cast<fftw_output_t *>(output.data()),
                           fftw_direction,
                           flags);
    };

    // REGULAR FFT 1D
    template <std::size_t dim, int fftw_direction, bool fftw_123dim, typename input_t, typename output_t, typename fftw_plan_dft_signature<input_t, output_t, dim, fftw_direction, fftw_123dim>::type fftw_plan_dft, bool half_plus_one_out, bool half_plus_one_in>
    inline auto fftw_plan_dft_caller(const xt::xarray<input_t, layout_type::row_major> &input, xt::xarray<output_t, layout_type::row_major> &output, unsigned int flags, bool /*odd_last_dim*/ = false)
    -> std::enable_if_t<dimensional::is_1<dim, fftw_123dim>::value && (fftw_direction != 0), typename fftw_t<input_t>::plan> {
      using fftw_input_t = fftw_number_t<input_t>;
      using fftw_output_t = fftw_number_t<output_t>;

      auto dft_dimensions_unsigned = dft_dimensions_from_output(output, half_plus_one_out);

      return fftw_plan_dft(static_cast<int>(dft_dimensions_unsigned[0]),
                           const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.data())),
                           reinterpret_cast<fftw_output_t *>(output.data()),
                           fftw_direction,
                           flags);
    };

    // REGULAR FFT 2D
    template <std::size_t dim, int fftw_direction, bool fftw_123dim, typename input_t, typename output_t, typename fftw_plan_dft_signature<input_t, output_t, dim, fftw_direction, fftw_123dim>::type fftw_plan_dft, bool half_plus_one_out, bool half_plus_one_in>
    inline auto fftw_plan_dft_caller(const xt::xarray<input_t, layout_type::row_major> &input, xt::xarray<output_t, layout_type::row_major> &output, unsigned int flags, bool /*odd_last_dim*/ = false)
    -> std::enable_if_t<dimensional::is_2<dim, fftw_123dim>::value && (fftw_direction != 0), typename fftw_t<input_t>::plan> {
      using fftw_input_t = fftw_number_t<input_t>;
      using fftw_output_t = fftw_number_t<output_t>;

      auto dft_dimensions_unsigned = dft_dimensions_from_output(output, half_plus_one_out);

      return fftw_plan_dft(static_cast<int>(dft_dimensions_unsigned[0]), static_cast<int>(dft_dimensions_unsigned[1]),
                           const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.data())),
                           reinterpret_cast<fftw_output_t *>(output.data()),
                           fftw_direction,
                           flags);
    };

    // REGULAR FFT 3D
    template <std::size_t dim, int fftw_direction, bool fftw_123dim, typename input_t, typename output_t, typename fftw_plan_dft_signature<input_t, output_t, dim, fftw_direction, fftw_123dim>::type fftw_plan_dft, bool half_plus_one_out, bool half_plus_one_in>
    inline auto fftw_plan_dft_caller(const xt::xarray<input_t, layout_type::row_major> &input, xt::xarray<output_t, layout_type::row_major> &output, unsigned int flags, bool /*odd_last_dim*/ = false)
    -> std::enable_if_t<dimensional::is_3<dim, fftw_123dim>::value && (fftw_direction != 0), typename fftw_t<input_t>::plan> {
      using fftw_input_t = fftw_number_t<input_t>;
      using fftw_output_t = fftw_number_t<output_t>;

      auto dft_dimensions_unsigned = dft_dimensions_from_output(output, half_plus_one_out);

      return fftw_plan_dft(static_cast<int>(dft_dimensions_unsigned[0]), static_cast<int>(dft_dimensions_unsigned[1]), static_cast<int>(dft_dimensions_unsigned[2]),
                           const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.data())),
                           reinterpret_cast<fftw_output_t *>(output.data()),
                           fftw_direction,
                           flags);
    };

    // REAL FFT N-dim
    template <std::size_t dim, int fftw_direction, bool fftw_123dim, typename input_t, typename output_t, typename fftw_plan_dft_signature<input_t, output_t, dim, 0, fftw_123dim>::type fftw_plan_dft, bool half_plus_one_out, bool half_plus_one_in>
    inline auto fftw_plan_dft_caller(const xt::xarray<input_t, layout_type::row_major> &input, xt::xarray<output_t, layout_type::row_major> &output, unsigned int flags, bool odd_last_dim = false)
    -> std::enable_if_t<dimensional::is_n<dim, fftw_123dim>::value && (fftw_direction == 0), typename fftw_t<input_t>::plan> {
      using fftw_input_t = fftw_number_t<input_t>;
      using fftw_output_t = fftw_number_t<output_t>;

      auto dft_dimensions_unsigned = dft_dimensions_from_output(output, half_plus_one_out, odd_last_dim);
      std::vector<int> dft_dimensions;
      dft_dimensions.reserve(dft_dimensions_unsigned.size());
      std::transform(dft_dimensions_unsigned.begin(), dft_dimensions_unsigned.end(), std::back_inserter(dft_dimensions), [&](std::size_t d) { return static_cast<int>(d); });

      return fftw_plan_dft(static_cast<int>(dim), dft_dimensions.data(),
                           const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.data())),
                           reinterpret_cast<fftw_output_t *>(output.data()),
                           flags);
    };

    // REAL FFT 1D
    template <std::size_t dim, int fftw_direction, bool fftw_123dim, typename input_t, typename output_t, typename fftw_plan_dft_signature<input_t, output_t, dim, 0, fftw_123dim>::type fftw_plan_dft, bool half_plus_one_out, bool half_plus_one_in>
    inline auto fftw_plan_dft_caller(const xt::xarray<input_t, layout_type::row_major> &input, xt::xarray<output_t, layout_type::row_major> &output, unsigned int flags, bool odd_last_dim = false)
    -> std::enable_if_t<dimensional::is_1<dim, fftw_123dim>::value && (fftw_direction == 0), typename fftw_t<input_t>::plan> {
      using fftw_input_t = fftw_number_t<input_t>;
      using fftw_output_t = fftw_number_t<output_t>;

      auto dft_dimensions_unsigned = dft_dimensions_from_output(output, half_plus_one_out, odd_last_dim);

      return fftw_plan_dft(static_cast<int>(dft_dimensions_unsigned[0]),
               const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.data())),
               reinterpret_cast<fftw_output_t *>(output.data()),
               flags);
    };

    // REAL FFT 2D
    template <std::size_t dim, int fftw_direction, bool fftw_123dim, typename input_t, typename output_t, typename fftw_plan_dft_signature<input_t, output_t, dim, 0, fftw_123dim>::type fftw_plan_dft, bool half_plus_one_out, bool half_plus_one_in>
    inline auto fftw_plan_dft_caller(const xt::xarray<input_t, layout_type::row_major> &input, xt::xarray<output_t, layout_type::row_major> &output, unsigned int flags, bool odd_last_dim = false)
    -> std::enable_if_t<dimensional::is_2<dim, fftw_123dim>::value && (fftw_direction == 0), typename fftw_t<input_t>::plan> {
      using fftw_input_t = fftw_number_t<input_t>;
      using fftw_output_t = fftw_number_t<output_t>;

      auto dft_dimensions_unsigned = dft_dimensions_from_output(output, half_plus_one_out, odd_last_dim);

      return fftw_plan_dft(static_cast<int>(dft_dimensions_unsigned[0]), static_cast<int>(dft_dimensions_unsigned[1]),
          const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.data())),
          reinterpret_cast<fftw_output_t *>(output.data()),
          flags);
    };

    // REAL FFT 3D
    template <std::size_t dim, int fftw_direction, bool fftw_123dim, typename input_t, typename output_t, typename fftw_plan_dft_signature<input_t, output_t, dim, 0, fftw_123dim>::type fftw_plan_dft, bool half_plus_one_out, bool half_plus_one_in>
    inline auto fftw_plan_dft_caller(const xt::xarray<input_t, layout_type::row_major> &input, xt::xarray<output_t, layout_type::row_major> &output, unsigned int flags, bool odd_last_dim = false)
    -> std::enable_if_t<dimensional::is_3<dim, fftw_123dim>::value && (fftw_direction == 0), typename fftw_t<input_t>::plan> {
      using fftw_input_t = fftw_number_t<input_t>;
      using fftw_output_t = fftw_number_t<output_t>;

      auto dft_dimensions_unsigned = dft_dimensions_from_output(output, half_plus_one_out, odd_last_dim);

      return fftw_plan_dft(static_cast<int>(dft_dimensions_unsigned[0]), static_cast<int>(dft_dimensions_unsigned[1]), static_cast<int>(dft_dimensions_unsigned[2]),
          const_cast<fftw_input_t *>(reinterpret_cast<const fftw_input_t *>(input.data())),
          reinterpret_cast<fftw_output_t *>(output.data()),
          flags);
    };


    ////
    // General: xarray templates
    ////

//    template<typename input_t, typename output_t, typename...>
//    inline xt::xarray<output_t> _fft_ (const xt::xarray<input_t, layout_type::row_major> &input) {
//      static_assert(sizeof(prec_t<input_t>) == 0, "Only specializations of _fft_ can be used");
//    }
//
//    template<typename input_t, typename output_t, typename...>
//    inline xt::xarray<output_t> _ifft_ (const xt::xarray<input_t, layout_type::row_major> &input) {
//      static_assert(sizeof(prec_t<input_t>) == 0, "Only specializations of _ifft_ can be used");
//    }

    template <
        typename input_t, typename output_t, std::size_t dim, int fftw_direction, bool fftw_123dim, bool half_plus_one_out, bool half_plus_one_in,
        typename fftw_plan_dft_signature<input_t, output_t, dim, fftw_direction, fftw_123dim>::type fftw_plan_dft,
        void (&fftw_execute)(typename fftw_t<input_t>::plan), void (&fftw_destroy_plan)(typename fftw_t<input_t>::plan),
        typename = std::enable_if_t<
            std::is_same< prec_t<input_t>, prec_t<output_t> >::value  // input and output precision must be the same
            && std::is_floating_point< prec_t<input_t> >::value       // numbers must be float, double or long double
            && (dimensional::is_123<dim, fftw_123dim>::value          // dimensionality must match fftw_123dim
                || dimensional::is_n<dim, fftw_123dim>::value)
        >
    >
    inline xt::xarray<output_t> _fft_(const xt::xarray<input_t, layout_type::row_major> &input) {
      auto output_shape = output_shape_from_input(input, half_plus_one_out, half_plus_one_in, false);
      xt::xarray<output_t, layout_type::row_major> output(output_shape);

      bool odd_last_dim = (input.shape()[input.shape().size()-1] % 2 != 0);

      auto plan = fftw_plan_dft_caller<dim, fftw_direction, fftw_123dim, input_t, output_t, fftw_plan_dft, half_plus_one_out, half_plus_one_in>(input, output, FFTW_ESTIMATE, odd_last_dim);
      if (plan == nullptr) {
        throw std::runtime_error("Plan creation returned nullptr. This usually means FFTW cannot create a plan for the given arguments (e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
      }

      fftw_execute(plan);
      fftw_destroy_plan(plan);
      return output;
    };

    template <
        typename input_t, typename output_t, std::size_t dim, int fftw_direction, bool fftw_123dim, bool half_plus_one_out, bool half_plus_one_in,
        typename fftw_plan_dft_signature<input_t, output_t, dim, fftw_direction, fftw_123dim>::type fftw_plan_dft,
        void (&fftw_execute)(typename fftw_t<input_t>::plan), void (&fftw_destroy_plan)(typename fftw_t<input_t>::plan),
        typename = std::enable_if_t<
            std::is_same< prec_t<input_t>, prec_t<output_t> >::value  // input and output precision must be the same
            && std::is_floating_point< prec_t<input_t> >::value       // numbers must be float, double or long double
            && (dimensional::is_123<dim, fftw_123dim>::value          // dimensionality must match fftw_123dim
                || dimensional::is_n<dim, fftw_123dim>::value)
        >
    >
    inline xt::xarray<output_t> _ifft_(const xt::xarray<input_t, layout_type::row_major> &input, bool odd_last_dim = false) {
      auto output_shape = output_shape_from_input(input, half_plus_one_out, half_plus_one_in, odd_last_dim);
      xt::xarray<output_t, layout_type::row_major> output(output_shape);

      auto plan = fftw_plan_dft_caller<dim, fftw_direction, fftw_123dim, input_t, output_t, fftw_plan_dft, half_plus_one_out, half_plus_one_in>(input, output, FFTW_ESTIMATE, odd_last_dim);
      if (plan == nullptr) {
        throw std::runtime_error("Plan creation returned nullptr. This usually means FFTW cannot create a plan for the given arguments (e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
      }

      fftw_execute(plan);
      fftw_destroy_plan(plan);
      auto dft_dimensions = dft_dimensions_from_output(output, half_plus_one_out, odd_last_dim);
      auto N_dft = static_cast<prec_t<output_t> >(std::accumulate(dft_dimensions.begin(), dft_dimensions.end(), static_cast<size_t>(1u), std::multiplies<size_t>()));
      return output / N_dft;
    };

    template <
        typename input_t, typename output_t, std::size_t dim, int fftw_direction, bool fftw_123dim, bool half_plus_one_out, bool half_plus_one_in,
        typename fftw_plan_dft_signature<input_t, output_t, dim, fftw_direction, fftw_123dim>::type fftw_plan_dft,
        void (&fftw_execute)(typename fftw_t<input_t>::plan), void (&fftw_destroy_plan)(typename fftw_t<input_t>::plan),
        typename = std::enable_if_t<
            std::is_same< prec_t<input_t>, prec_t<output_t> >::value  // input and output precision must be the same
            && std::is_floating_point< prec_t<input_t> >::value       // numbers must be float, double or long double
            && (dimensional::is_123<dim, fftw_123dim>::value          // dimensionality must match fftw_123dim
                || dimensional::is_n<dim, fftw_123dim>::value)
        >
    >
    inline xt::xarray<output_t> _hfft_(const xt::xarray<input_t, layout_type::row_major> &input) {
      auto output_shape = output_shape_from_input(input, half_plus_one_out, half_plus_one_in);
      xt::xarray<output_t, layout_type::row_major> output(output_shape);

      xt::xarray<input_t, layout_type::row_major> input_conj = xt::conj(input);

      auto plan = fftw_plan_dft_caller<dim, fftw_direction, fftw_123dim, input_t, output_t, fftw_plan_dft, half_plus_one_out, half_plus_one_in>(input_conj, output, FFTW_ESTIMATE);
      if (plan == nullptr) {
        throw std::runtime_error("Plan creation returned nullptr. This usually means FFTW cannot create a plan for the given arguments (e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
      }

      fftw_execute(plan);
      fftw_destroy_plan(plan);
      return output;
    };

    template <
        typename input_t, typename output_t, std::size_t dim, int fftw_direction, bool fftw_123dim, bool half_plus_one_out, bool half_plus_one_in,
        typename fftw_plan_dft_signature<input_t, output_t, dim, fftw_direction, fftw_123dim>::type fftw_plan_dft,
        void (&fftw_execute)(typename fftw_t<input_t>::plan), void (&fftw_destroy_plan)(typename fftw_t<input_t>::plan),
        typename = std::enable_if_t<
            std::is_same< prec_t<input_t>, prec_t<output_t> >::value  // input and output precision must be the same
            && std::is_floating_point< prec_t<input_t> >::value       // numbers must be float, double or long double
            && (dimensional::is_123<dim, fftw_123dim>::value          // dimensionality must match fftw_123dim
                || dimensional::is_n<dim, fftw_123dim>::value)
        >
    >
    inline xt::xarray<output_t> _ihfft_(const xt::xarray<input_t, layout_type::row_major> &input) {
      auto output_shape = output_shape_from_input(input, half_plus_one_out, half_plus_one_in);
      xt::xarray<output_t, layout_type::row_major> output(output_shape);

      auto plan = fftw_plan_dft_caller<dim, fftw_direction, fftw_123dim, input_t, output_t, fftw_plan_dft, half_plus_one_out, half_plus_one_in>(input, output, FFTW_ESTIMATE);
      if (plan == nullptr) {
        throw std::runtime_error("Plan creation returned nullptr. This usually means FFTW cannot create a plan for the given arguments (e.g. a non-destructive multi-dimensional real FFT is impossible in FFTW).");
      }

      fftw_execute(plan);
      fftw_destroy_plan(plan);

      output = xt::conj(output);

      auto dft_dimensions = dft_dimensions_from_output(output, half_plus_one_out);
      auto N_dft = static_cast<prec_t<output_t> >(std::accumulate(dft_dimensions.begin(), dft_dimensions.end(), static_cast<size_t>(1u), std::multiplies<size_t>()));
      return output / N_dft;
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
//                                              const_cast<real_t *>(input.data()),
//                                              reinterpret_cast<fftwXXXXXXX_complex*>(output.data()),
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
//                                                      const_cast<fftwXXXXX_complex *>(reinterpret_cast<const fftwXXXXX_complex *>(input.data())),
//                                                      output.data(),
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

    inline xt::xarray<std::complex<float> > fft (const xt::xarray<std::complex<float> > &input) {
      return _fft_<std::complex<float>, std::complex<float>, 1, FFTW_FORWARD, true, false, false, fftwf_plan_dft_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<float> > ifft (const xt::xarray<std::complex<float> > &input) {
      return _ifft_<std::complex<float>, std::complex<float>, 1, FFTW_BACKWARD, true, false, false, fftwf_plan_dft_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > fft (const xt::xarray<std::complex<double> > &input) {
      return _fft_<std::complex<double>, std::complex<double>, 1, FFTW_FORWARD, true, false, false, fftw_plan_dft_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > ifft (const xt::xarray<std::complex<double> > &input) {
      return _ifft_<std::complex<double>, std::complex<double>, 1, FFTW_BACKWARD, true, false, false, fftw_plan_dft_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > fft (const xt::xarray<std::complex<long double> > &input) {
      return _fft_<std::complex<long double>, std::complex<long double>, 1, FFTW_FORWARD, true, false, false, fftwl_plan_dft_1d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > ifft (const xt::xarray<std::complex<long double> > &input) {
      return _ifft_<std::complex<long double>, std::complex<long double>, 1, FFTW_BACKWARD, true, false, false, fftwl_plan_dft_1d, fftwl_execute, fftwl_destroy_plan> (input);
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

    inline xt::xarray<std::complex<double> > fft2 (const xt::xarray<std::complex<double> > &input) {
      return _fft_<std::complex<double>, std::complex<double>, 2, FFTW_FORWARD, true, false, false, fftw_plan_dft_2d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > ifft2 (const xt::xarray<std::complex<double> > &input) {
      return _ifft_<std::complex<double>, std::complex<double>, 2, FFTW_BACKWARD, true, false, false, fftw_plan_dft_2d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > fft2 (const xt::xarray<std::complex<long double> > &input) {
      return _fft_<std::complex<long double>, std::complex<long double>, 2, FFTW_FORWARD, true, false, false, fftwl_plan_dft_2d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > ifft2 (const xt::xarray<std::complex<long double> > &input) {
      return _ifft_<std::complex<long double>, std::complex<long double>, 2, FFTW_BACKWARD, true, false, false, fftwl_plan_dft_2d, fftwl_execute, fftwl_destroy_plan> (input);
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

    inline xt::xarray<std::complex<double> > fft3 (const xt::xarray<std::complex<double> > &input) {
      return _fft_<std::complex<double>, std::complex<double>, 3, FFTW_FORWARD, true, false, false, fftw_plan_dft_3d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > ifft3 (const xt::xarray<std::complex<double> > &input) {
      return _ifft_<std::complex<double>, std::complex<double>, 3, FFTW_BACKWARD, true, false, false, fftw_plan_dft_3d, fftw_execute, fftw_destroy_plan> (input);
    }

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
    inline xt::xarray<std::complex<float> > fftn (const xt::xarray<std::complex<float> > &input) {
      return _fft_<std::complex<float>, std::complex<float>, dim, FFTW_FORWARD, false, false, false, fftwf_plan_dft, fftwf_execute, fftwf_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<std::complex<float> > ifftn (const xt::xarray<std::complex<float> > &input) {
      return _ifft_<std::complex<float>, std::complex<float>, dim, FFTW_BACKWARD, false, false, false, fftwf_plan_dft, fftwf_execute, fftwf_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<std::complex<double> > fftn (const xt::xarray<std::complex<double> > &input) {
      return _fft_<std::complex<double>, std::complex<double>, dim, FFTW_FORWARD, false, false, false, fftw_plan_dft, fftw_execute, fftw_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<std::complex<double> > ifftn (const xt::xarray<std::complex<double> > &input) {
      return _ifft_<std::complex<double>, std::complex<double>, dim, FFTW_BACKWARD, false, false, false, fftw_plan_dft, fftw_execute, fftw_destroy_plan> (input);
    }

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

    inline xt::xarray<std::complex<float> > rfft (const xt::xarray<float> &input) {
      return _fft_<float, std::complex<float>, 1, 0, true, true, false, fftwf_plan_dft_r2c_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<float> irfft (const xt::xarray<std::complex<float> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<float>, float, 1, 0, true, false, true, fftwf_plan_dft_c2r_1d, fftwf_execute, fftwf_destroy_plan> (input, odd_last_dim);
    }

    inline xt::xarray<std::complex<double> > rfft (const xt::xarray<double> &input) {
      return _fft_<double, std::complex<double>, 1, 0, true, true, false, fftw_plan_dft_r2c_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<double> irfft (const xt::xarray<std::complex<double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<double>, double, 1, 0, true, false, true, fftw_plan_dft_c2r_1d, fftw_execute, fftw_destroy_plan> (input, odd_last_dim);
    }

    inline xt::xarray<std::complex<long double> > rfft (const xt::xarray<long double> &input) {
      return _fft_<long double, std::complex<long double>, 1, 0, true, true, false, fftwl_plan_dft_r2c_1d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<long double> irfft (const xt::xarray<std::complex<long double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<long double>, long double, 1, 0, true, false, true, fftwl_plan_dft_c2r_1d, fftwl_execute, fftwl_destroy_plan> (input, odd_last_dim);
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

    inline xt::xarray<std::complex<double> > rfft2 (const xt::xarray<double> &input) {
      return _fft_<double, std::complex<double>, 2, 0, true, true, false, fftw_plan_dft_r2c_2d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<double> irfft2 (const xt::xarray<std::complex<double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<double>, double, 2, 0, true, false, true, fftw_plan_dft_c2r_2d, fftw_execute, fftw_destroy_plan> (input, odd_last_dim);
    }

    inline xt::xarray<std::complex<long double> > rfft2 (const xt::xarray<long double> &input) {
      return _fft_<long double, std::complex<long double>, 2, 0, true, true, false, fftwl_plan_dft_r2c_2d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<long double> irfft2 (const xt::xarray<std::complex<long double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<long double>, long double, 2, 0, true, false, true, fftwl_plan_dft_c2r_2d, fftwl_execute, fftwl_destroy_plan> (input, odd_last_dim);
    }


    ////
    // Real FFT: 3D
    ////

    // this doesn't work because the plan function is also precision dependent and it would be madness to include them all in fftw_t...
//    template<typename input_t, typename output_t>
//    constexpr auto rfft3 = _fft_<input_t, std::complex<output_t>, 3, true, false, fftwl_plan_dft_r2c_3d, int, int, int>;
//    template<typename input_t, typename output_t>
//    constexpr auto irfft3 = _ifft_<input_t, std::complex<output_t>, 3, true, false, fftwl_plan_dft_r2c_3d, int, int, int>;

    inline xt::xarray<std::complex<float> > rfft3 (const xt::xarray<float> &input) {
      return _fft_<float, std::complex<float>, 3, 0, true, true, false, fftwf_plan_dft_r2c_3d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<float> irfft3 (const xt::xarray<std::complex<float> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<float>, float, 3, 0, true, false, true, fftwf_plan_dft_c2r_3d, fftwf_execute, fftwf_destroy_plan> (input, odd_last_dim);
    }

    inline xt::xarray<std::complex<double> > rfft3 (const xt::xarray<double> &input) {
      return _fft_<double, std::complex<double>, 3, 0, true, true, false, fftw_plan_dft_r2c_3d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<double> irfft3 (const xt::xarray<std::complex<double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<double>, double, 3, 0, true, false, true, fftw_plan_dft_c2r_3d, fftw_execute, fftw_destroy_plan> (input, odd_last_dim);
    }

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
    inline xt::xarray<std::complex<float> > rfftn (const xt::xarray<float> &input) {
      return _fft_<float, std::complex<float>, dim, 0, false, true, false, fftwf_plan_dft_r2c, fftwf_execute, fftwf_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<float> irfftn (const xt::xarray<std::complex<float> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<float>, float, dim, 0, false, false, true, fftwf_plan_dft_c2r, fftwf_execute, fftwf_destroy_plan> (input, odd_last_dim);
    }

    template <std::size_t dim>
    inline xt::xarray<std::complex<double> > rfftn (const xt::xarray<double> &input) {
      return _fft_<double, std::complex<double>, dim, 0, false, true, false, fftw_plan_dft_r2c, fftw_execute, fftw_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<double> irfftn (const xt::xarray<std::complex<double> > &input, bool odd_last_dim = false) {
      return _ifft_<std::complex<double>, double, dim, 0, false, false, true, fftw_plan_dft_c2r, fftw_execute, fftw_destroy_plan> (input, odd_last_dim);
    }

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

    inline xt::xarray<float> hfft (const xt::xarray<std::complex<float> > &input) {
      return _hfft_<std::complex<float>, float, 1, 0, true, false, true, fftwf_plan_dft_c2r_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<float> > ihfft (const xt::xarray<float> &input) {
      return _ihfft_<float, std::complex<float>, 1, 0, true, true, false, fftwf_plan_dft_r2c_1d, fftwf_execute, fftwf_destroy_plan> (input);
    }

    inline xt::xarray<double> hfft (const xt::xarray<std::complex<double> > &input) {
      return _hfft_<std::complex<double>, double, 1, 0, true, false, true, fftw_plan_dft_c2r_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > ihfft (const xt::xarray<double> &input) {
      return _ihfft_<double, std::complex<double>, 1, 0, true, true, false, fftw_plan_dft_r2c_1d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<long double> hfft (const xt::xarray<std::complex<long double> > &input) {
      return _hfft_<std::complex<long double>, long double, 1, 0, true, false, true, fftwl_plan_dft_c2r_1d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > ihfft (const xt::xarray<long double> &input) {
      return _ihfft_<long double, std::complex<long double>, 1, 0, true, true, false, fftwl_plan_dft_r2c_1d, fftwl_execute, fftwl_destroy_plan> (input);
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

    inline xt::xarray<double> hfft2 (const xt::xarray<std::complex<double> > &input) {
      return _hfft_<std::complex<double>, double, 2, 0, true, false, true, fftw_plan_dft_c2r_2d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > ihfft2 (const xt::xarray<double> &input) {
      return _ihfft_<double, std::complex<double>, 2, 0, true, true, false, fftw_plan_dft_r2c_2d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<long double> hfft2 (const xt::xarray<std::complex<long double> > &input) {
      return _hfft_<std::complex<long double>, long double, 2, 0, true, false, true, fftwl_plan_dft_c2r_2d, fftwl_execute, fftwl_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<long double> > ihfft2 (const xt::xarray<long double> &input) {
      return _ihfft_<long double, std::complex<long double>, 2, 0, true, true, false, fftwl_plan_dft_r2c_2d, fftwl_execute, fftwl_destroy_plan> (input);
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

    inline xt::xarray<double> hfft3 (const xt::xarray<std::complex<double> > &input) {
      return _hfft_<std::complex<double>, double, 3, 0, true, false, true, fftw_plan_dft_c2r_3d, fftw_execute, fftw_destroy_plan> (input);
    }

    inline xt::xarray<std::complex<double> > ihfft3 (const xt::xarray<double> &input) {
      return _ihfft_<double, std::complex<double>, 3, 0, true, true, false, fftw_plan_dft_r2c_3d, fftw_execute, fftw_destroy_plan> (input);
    }

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
    inline xt::xarray<float> hfftn (const xt::xarray<std::complex<float> > &input) {
      return _hfft_<std::complex<float>, float, dim, 0, false, false, true, fftwf_plan_dft_c2r, fftwf_execute, fftwf_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<std::complex<float> > ihfftn (const xt::xarray<float> &input) {
      return _ihfft_<float, std::complex<float>, dim, 0, false, true, false, fftwf_plan_dft_r2c, fftwf_execute, fftwf_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<double> hfftn (const xt::xarray<std::complex<double> > &input) {
      return _hfft_<std::complex<double>, double, dim, 0, false, false, true, fftw_plan_dft_c2r, fftw_execute, fftw_destroy_plan> (input);
    }

    template <std::size_t dim>
    inline xt::xarray<std::complex<double> > ihfftn (const xt::xarray<double> &input) {
      return _ihfft_<double, std::complex<double>, dim, 0, false, true, false, fftw_plan_dft_r2c, fftw_execute, fftw_destroy_plan> (input);
    }

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

#endif //XTENSOR_FFTW_BASIC_HPP
