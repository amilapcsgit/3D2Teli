@echo off
setlocal

cd /d "%~dp0"

echo Launching Tent-Maker Pro (Qt-first, Tk fallback)...
set "TENTMAKER_FORCE_TK=0"
python main.py
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo Tent-Maker Pro exited with code %EXIT_CODE%.
)

exit /b %EXIT_CODE%
