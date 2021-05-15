/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 *
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 *
 * basic_option.hpp:
 * Include xtensor-fftw functionalities with explicit defined precision types.
 *
 */

#ifndef XTENSOR_FFTW_BASIC_OPTION_HPP
#define XTENSOR_FFTW_BASIC_OPTION_HPP

#if !defined(XTENSOR_FFTW_USE_FLOAT) \
    && !defined(XTENSOR_FFTW_USE_DOUBLE) \
    && !defined(XTENSOR_FFTW_USE_LONG_DOUBLE)
#error Missing definition of FFTW type library. Please #define at least once before include xtensor-fftw library.
#endif

#ifdef XTENSOR_FFTW_USE_FLOAT
#include "xtensor-fftw/basic_float.hpp"
#endif

#ifdef XTENSOR_FFTW_USE_DOUBLE
#include "xtensor-fftw/basic_double.hpp"
#endif

#ifdef XTENSOR_FFTW_USE_LONG_DOUBLE
#include "xtensor-fftw/basic_long_double.hpp"
#endif

#endif //XTENSOR_FFTW_BASIC_OPTION_HPP
