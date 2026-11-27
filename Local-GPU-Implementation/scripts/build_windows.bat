@echo off
REM Build (and optionally run) the Parallel Reduction project on Windows.
REM Usage:  scripts\build_windows.bat          -> build only
REM         scripts\build_windows.bat run      -> build, then run the full benchmark
REM Needs: CUDA Toolkit + Visual Studio "Desktop development with C++".
REM Easiest: run from "x64 Native Tools Command Prompt for VS".
setlocal EnableDelayedExpansion
cd /d "%~dp0.."
if not exist build mkdir build
if not exist results mkdir results

where nvcc >nul 2>nul
if errorlevel 1 (
  echo ERROR: nvcc not found. Install the CUDA Toolkit and reopen the terminal.
  exit /b 1
)

REM nvcc needs the MSVC compiler cl.exe. Try to load it automatically if missing.
where cl >nul 2>nul
if errorlevel 1 (
  set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
  if exist "!VSWHERE!" (
    for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
    if defined VSPATH call "!VSPATH!\VC\Auxiliary\Build\vcvars64.bat" >nul
  )
)
where cl >nul 2>nul
if errorlevel 1 (
  echo ERROR: cl.exe not found. Open "x64 Native Tools Command Prompt for VS" and run this script again.
  exit /b 1
)

nvcc --version
set "SRC=src\parallel_reduction.cu"
set "OUT=build\parallel_reduction.exe"
set "FLAGS=-O3 -std=c++17 -lineinfo"

REM 1) Let nvcc detect the GPU.
nvcc %FLAGS% -arch=native %SRC% -o %OUT% 2>nul
if not errorlevel 1 (
  echo Built with -arch=native
  goto built
)
REM 2) Ask the driver for the compute capability, e.g. 8.6 -> 86.
set "CC="
for /f "tokens=*" %%c in ('nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul') do if not defined CC set "CC=%%c"
if defined CC (
  set "CC=!CC:.=!"
  set "CC=!CC: =!"
  echo Building for sm_!CC!
  nvcc %FLAGS% -gencode arch=compute_!CC!,code=[sm_!CC!,compute_!CC!] %SRC% -o %OUT%
) else (
  echo Could not detect GPU architecture; using nvcc default target.
  nvcc %FLAGS% %SRC% -o %OUT%
)
if errorlevel 1 (
  echo BUILD FAILED
  exit /b 1
)

:built
echo Build OK -^> %OUT%
if /i "%~1"=="run" (
  %OUT% > results\run_log.txt
  type results\run_log.txt
)
endlocal
