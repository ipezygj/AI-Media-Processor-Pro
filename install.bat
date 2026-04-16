@echo off
title AI Media Processor Pro - Installer
echo ============================================
echo  AI Media Processor Pro - Asennus
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [VIRHE] Python ei loytynyt. Asenna Python 3.10+ ja lisaa PATH:iin.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python loydetty.

:: Check FFmpeg
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo [VAROITUS] FFmpeg ei loytynyt. Yritetaan asentaa winget:lla...
    winget install ffmpeg >nul 2>&1
    if %errorlevel% neq 0 (
        echo [VIRHE] FFmpeg-asennus epaonnistui. Asenna manuaalisesti:
        echo https://ffmpeg.org/download.html
        pause
        exit /b 1
    )
    echo [OK] FFmpeg asennettu.
) else (
    echo [OK] FFmpeg loydetty.
)

:: Check NVIDIA GPU
echo.
echo Tarkistetaan GPU-tuki...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU havaittu - asennetaan CUDA-tuella.
    pip install --no-cache-dir torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [INFO] NVIDIA GPU:ta ei havaittu - asennetaan CPU-versio.
    pip install --no-cache-dir torch torchaudio torchvision
)

:: Install remaining dependencies
echo.
echo Asennetaan muut riippuvuudet...
pip install --no-cache-dir -r "%~dp0requirements.txt"

echo.
echo ============================================
echo  Asennus valmis!
echo  Kaynnista: run.bat
echo ============================================
pause
