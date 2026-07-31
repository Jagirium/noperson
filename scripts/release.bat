@echo off
setlocal
echo Choose release build:
echo   3^) Windows GPU native
set /p NOPERSON_RELEASE_CHOICE=^> 
if "%NOPERSON_RELEASE_CHOICE%"=="3" call "%~dp0release\win.bat" %*
if not "%NOPERSON_RELEASE_CHOICE%"=="3" exit /b 2
exit /b %ERRORLEVEL%
