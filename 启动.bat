@echo off
REM ASCII only. Chinese text inside a .bat gets mangled because cmd reads the
REM file in the system ANSI codepage (GBK), which breaks quote pairing.
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
title Live Meeting Minutes

if not exist .venv goto mkvenv
goto activate

:mkvenv
echo Creating Python environment, first run only...
py -m venv .venv
if errorlevel 1 goto nopython

:activate
call .venv\Scripts\activate.bat

python -c "import PySide6" >nul 2>&1
if errorlevel 1 goto install
python -c "import pyaudiowpatch" >nul 2>&1
if errorlevel 1 goto install
python -c "import whisperlivekit" >nul 2>&1
if errorlevel 1 goto noasr
goto run

:install
echo Installing dependencies, this may take a few minutes...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 goto err
python -c "import whisperlivekit" >nul 2>&1
if errorlevel 1 goto noasr
goto run

:noasr
echo.
echo whisperlivekit is not installed. Run this once:
echo.
echo   .venv\Scripts\activate.bat
echo   pip install "whisperlivekit[qwen3-streaming]" -i https://pypi.tuna.tsinghua.edu.cn/simple
echo.
goto end

:run
echo.
echo Starting the desktop app.
echo Make sure LM Studio Local Server is running on port 1234.
echo Close the app window to exit. (Web console fallback: python app.py)
echo.
python desktop.py
goto end

:nopython
echo.
echo Python not found. Install Python 3.11 or newer and tick "Add to PATH".
echo https://www.python.org/downloads/
goto end

:err
echo.
echo Dependency install failed. Please send the error above.

:end
pause
