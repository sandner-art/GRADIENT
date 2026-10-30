@echo off
rem Start GRADIENT (API + web UI) and open the browser. Extra args go to `gradient serve`.
rem In PowerShell run it as .\run.bat
cd /d "%~dp0"
if not exist ".venv\Scripts\gradient.exe" (
  echo GRADIENT is not installed yet. Run install.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\gradient.exe" serve --open %*
if errorlevel 1 (
  echo.
  echo GRADIENT stopped with an error - see the messages above.
  pause
)
