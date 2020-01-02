'''
Before use, install releash:
 * $ python -m pip install releash
 * Install hub (to do a PR from the command line) https://hub.github.com/
Workflow:
 * releash status
 * releash bump -i -v
 * releash release -i -v
Lose the -i (interactive) and -v (verbose) when comfortable
'''

from releash import *
gitpush = ReleaseTargetGitPush()

xfftw = add_package(path=".", name="xtensor-fftw")
version_xfftw = VersionSourceAndTargetHpp(xfftw, '{path}/include/xtensor-fftw/xtensor-fftw_config.hpp', prefix='XTENSOR_FFTW_VERSION_')
gittag_xfftw = ReleaseTargetGitTagVersion(version_source=version_xfftw, prefix='', annotate=True, msg=None)

xfftw.version_source = version_xfftw

xfftw.version_targets.append(version_xfftw)
xfftw.release_targets.append(gittag_xfftw)
xfftw.release_targets.append(gitpush)

source_tarball_filename = 'https://github.com/xtensor-stack/xtensor-fftw/archive/{version}.tar.gz'.format(version=version_xfftw)
conda_forge_xfftw = ReleaseTargetCondaForge(xfftw, '../xtensor-fftw-feedstock', source_tarball_filename=source_tarball_filename)
xfftw.release_targets.append(conda_forge_xfftw)
