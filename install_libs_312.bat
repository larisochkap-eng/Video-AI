@echo off
REM Повністю перенаправляємо системні тимчасові папки на диск E:
set "TEMP=E:\VideoAi2\.temp"
set "TMP=E:\VideoAi2\.temp"
set "PIP_CACHE_DIR=E:\VideoAi2\.pip_cache"

REM Створюємо ці папки, якщо їх ще немає
if not exist "E:\VideoAi2\.temp" mkdir "E:\VideoAi2\.temp"
if not exist "E:\VideoAi2\.pip_cache" mkdir "E:\VideoAi2\.pip_cache"

set PY=py -3.12

cd /d "%~dp0"
echo Folder: %CD%
%PY% --version
if errorlevel 1 goto nopy
echo.

echo [1/2] Installing main libraries...
%PY% -m pip install -U --target . diffusers transformers accelerate sentencepiece imageio imageio-ffmpeg deep-translator bitsandbytes
if errorlevel 1 goto error

echo.
echo [2/2] Installing torch with CUDA (for NVIDIA GPU)...
%PY% -m pip install -U --target . torch --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto error

echo.
echo DONE. Now run start_video.bat
pause
exit /b 0

:nopy
echo Python 3.12 not found. Check with:  py -0
pause
exit /b 1

:error
echo.
echo ERROR during installation. Copy the text above and send it to Claude.
pause
exit /b 1