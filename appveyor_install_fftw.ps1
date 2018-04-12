mkdir $env:FFTW_ROOT
if ($env:Platform -eq "x86") {
  $source = "ftp://ftp.fftw.org/pub/fftw/fftw-3.3.5-dll32.zip"
  $destination = "C:\fftw-3.3.5.zip"
  Invoke-WebRequest $source -OutFile $destination
  echo "x86 fftw download complete"
}
if ($env:Platform -eq "x64") {
  $source = "ftp://ftp.fftw.org/pub/fftw/fftw-3.3.5-dll64.zip"
  $destination = "C:\fftw-3.3.5.zip"
  Invoke-WebRequest $source -OutFile $destination
  echo "x64 fftw download complete"
}

7z e C:\fftw-3.3.5.zip -o"${env:FFTW_ROOT}"
cd $env:FFTW_ROOT

if ($env:Platform -eq "x86") {
  lib.exe /def:libfftw3-3.def
  lib.exe /def:libfftw3f-3.def
  lib.exe /def:libfftw3l-3.def
}
if ($env:Platform -eq "x64") {
  lib.exe /machine:x64 /def:libfftw3-3.def
  lib.exe /machine:x64 /def:libfftw3f-3.def
  lib.exe /machine:x64 /def:libfftw3l-3.def
}

$env:PATH="${env:FFTW_ROOT};${env:PATH}"
