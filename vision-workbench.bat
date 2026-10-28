@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" %*
set "workbench_exit=%errorlevel%"
if not "%workbench_exit%"=="0" (
  echo.
  echo Startup failed. Read the error above and data\logs for details.
  pause
)
exit /b %workbench_exit%
