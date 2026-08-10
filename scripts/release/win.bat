@echo off
setlocal
set "NOPERSON_WIN_DEV="

:parse_args
if "%~1"=="" goto run_builder
if /I "%~1"=="--dev" goto select_dev
if /I "%~1"=="-Dev" goto select_dev
if /I "%~1"=="--orchestrated" goto discard_internal_marker
if /I "%~1"=="-Orchestrated" goto discard_internal_marker
echo release: unknown Windows release argument: %~1 1>&2
exit /b 2

:select_dev
set "NOPERSON_WIN_DEV=-Dev"
shift
goto parse_args

:discard_internal_marker
shift
goto parse_args

:run_builder
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0win.ps1" -Orchestrated %NOPERSON_WIN_DEV%
exit /b %ERRORLEVEL%
