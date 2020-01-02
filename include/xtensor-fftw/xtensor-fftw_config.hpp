/*
 * xtensor-fftw
 * Copyright (c) 2017, Patrick Bos
 * Distributed under the terms of the BSD 3-Clause License.
 *
 * The full license is in the file LICENSE, distributed with this software.
 */

#ifndef XTENSOR_FFTW_CONFIG_HPP
#define XTENSOR_FFTW_CONFIG_HPP

#define XTENSOR_FFTW_VERSION_MAJOR 0
#define XTENSOR_FFTW_VERSION_MINOR 2
#define XTENSOR_FFTW_VERSION_PATCH 6

// Define if the library is going to be using exceptions.
#if (!defined(__cpp_exceptions) && !defined(__EXCEPTIONS) && !defined(_CPPUNWIND))
#undef XTENSOR_FFTW_DISABLE_EXCEPTIONS
#define XTENSOR_FFTW_DISABLE_EXCEPTIONS
#endif

// Exception support.
#if defined(XTENSOR_FFTW_DISABLE_EXCEPTIONS)
#include <iostream>
#define XTENSOR_FFTW_THROW(_, msg)       \
    {                                    \
      std::cerr << msg << std::endl;     \
      std::abort();                      \
    }
#else
#define XTENSOR_FFTW_THROW(exception, msg) throw exception(msg)
#endif

#endif //XTENSOR_FFTW_CONFIG_HPP
