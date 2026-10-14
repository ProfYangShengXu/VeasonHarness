@echo off
rem Windows shim: `veason index video.mp4` from cmd/PowerShell.
setlocal
set "PY=%VEASON_PYTHON%"
if "%PY%"=="" set "PY=python"
"%PY%" "%~dp0veason.py" %*
exit /b %ERRORLEVEL%
