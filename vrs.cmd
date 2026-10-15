@echo off
rem Windows shim: `vrs index video.mp4` from cmd/PowerShell.
setlocal
set "PY=%VRS_PYTHON%"
if "%PY%"=="" set "PY=python"
"%PY%" "%~dp0vrs.py" %*
exit /b %ERRORLEVEL%
