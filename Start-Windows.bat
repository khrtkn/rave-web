@echo off
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
  py -3 launch.py
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python 3.9 or later is required. Install it from https://www.python.org/downloads/ .
    pause
    exit /b 1
  )
  python launch.py
)
if errorlevel 1 pause
