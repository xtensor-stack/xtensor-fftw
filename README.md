# xtensor-fftw
FFTW bindings for xtensor

## Installation

### Dependencies
/cmake/ and /xtensor/ must be installed in order to compile /xtensor-fftw/.
Both can either be installed through /conda/ or built/installed manually.
When using a non-/conda/ /xtensor/-install, make sure that the CMake `find_package` command can find /xtensor/, e.g. by passing something like ``-DCMAKE_MODULE_PATH="path_to_xtensorConfig.cmake"` to /cmake/ (not tested).

The main dependency is a version of FFTW 3.

### Configure, build and install

