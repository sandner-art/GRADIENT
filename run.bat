@echo off
rem Start GRADIENT (API + web UI) and open the browser. Extra args go to `gradient serve`.
cd /d "%~dp0"
if not exist ".venv\Scripts\gradient.exe" (
  echo GRADIENT is not installed yet. Run install.ps1 first.
  exit /b 1
)
".venv\Scripts\gradient.exe" serve --open %*
