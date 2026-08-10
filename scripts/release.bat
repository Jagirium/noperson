@echo off
setlocal
set "NOPERSON_RELEASE_CHOICE="
set "NOPERSON_RELEASE_DEV="

:parse_args
if "%~1"=="" goto choose_variant
if /I "%~1"=="--windows" goto select_windows
if /I "%~1"=="windows" goto select_windows
if "%~1"=="3" goto select_windows
if /I "%~1"=="--dev" goto select_dev
if /I "%~1"=="-Dev" goto select_dev
echo release: unknown release argument: %~1 1>&2
exit /b 2

:select_windows
if defined NOPERSON_RELEASE_CHOICE goto duplicate_variant
set "NOPERSON_RELEASE_CHOICE=3"
shift
goto parse_args

:duplicate_variant
echo release: choose exactly one release variant 1>&2
exit /b 2

:select_dev
set "NOPERSON_RELEASE_DEV=--dev"
shift
goto parse_args

:choose_variant
if defined NOPERSON_RELEASE_CHOICE goto run_platform_builder
echo Choose release build:
echo   3^) Windows GPU native
set /p NOPERSON_RELEASE_CHOICE=^> 

:run_platform_builder
if not "%NOPERSON_RELEASE_CHOICE%"=="3" exit /b 2
call "%~dp0release\win.bat" %NOPERSON_RELEASE_DEV%
exit /b %ERRORLEVEL%
