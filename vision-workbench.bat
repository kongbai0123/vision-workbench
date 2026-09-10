@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0bootstrap.ps1"
  if errorlevel 1 exit /b 1
)
start "Vision Workbench" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0main.py"
