@echo off
setlocal enabledelayedexpansion
title AgniAI Single-Click Installer and Launcher

:: Get absolute directory of script
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

echo ===================================================
echo             AgniAI 1-Click Installer ^& Launcher
echo ===================================================
echo.

:: 1. Check / Install Ollama
echo [1/5] Checking Ollama installation...
set "OLLAMA_EXE="
if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
) else (
    where ollama >nul 2>&1
    if !errorlevel! equ 0 (
        set "OLLAMA_EXE=ollama"
    )
)

if "%OLLAMA_EXE%"=="" (
    if exist "%ROOT_DIR%OllamaSetup.exe" (
        echo Ollama not found. Installing Ollama automatically from OllamaSetup.exe...
        start /wait "" "%ROOT_DIR%OllamaSetup.exe" /silent
        if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
            set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
        )
    ) else (
        echo [WARNING] Ollama is not installed and OllamaSetup.exe was not found.
        echo Please install Ollama from https://ollama.com before proceeding.
    )
) else (
    echo Ollama detected.
)

:: 2. Extract Ollama Model Folder (models.zip -> %USERPROFILE%\.ollama)
echo.
echo [2/5] Checking Ollama model weights...
set "OLLAMA_DIR=%USERPROFILE%\.ollama"
if not exist "%OLLAMA_DIR%" mkdir "%OLLAMA_DIR%"

if exist "%ROOT_DIR%models.zip" (
    echo Found Ollama models.zip. Extracting to %OLLAMA_DIR%...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%ROOT_DIR%models.zip' -DestinationPath '%OLLAMA_DIR%' -Force"
    echo Ollama model folder updated successfully!
) else if exist "%ROOT_DIR%models" (
    echo Copying local models directory to %OLLAMA_DIR%...
    xcopy /e /i /y /q "%ROOT_DIR%models" "%OLLAMA_DIR%\models" >nul
)

:: 3. Extract HuggingFace / SentenceTransformers Models (models--*.zip -> %USERPROFILE%\.cache\huggingface\hub)
echo.
echo [3/5] Checking HuggingFace model cache...
set "HF_CACHE=%USERPROFILE%\.cache\huggingface\hub"
if not exist "%HF_CACHE%" mkdir "%HF_CACHE%"

for %%F in ("%ROOT_DIR%models--*.zip") do (
    echo Extracting %%~nxF to HuggingFace cache...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%%F' -DestinationPath '%HF_CACHE%' -Force"
)

:: 4. Check / Copy .env file
echo.
echo [4/5] Checking configuration (.env)...
if not exist "%ROOT_DIR%.env" (
    if exist "%ROOT_DIR%.env.example" (
        echo Creating .env from .env.example...
        copy /y "%ROOT_DIR%.env.example" "%ROOT_DIR%.env" >nul
    ) else (
        echo [WARNING] .env file missing.
    )
) else (
    echo Configuration file (.env) found.
)

:: 5. Start Ollama Service & Launch AgniAI Application
echo.
echo [5/5] Starting services...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Process -Name ollama -ErrorAction SilentlyContinue" >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Ollama service...
    if not "%OLLAMA_EXE%"=="" (
        start "" /b "%OLLAMA_EXE%" serve >nul 2>&1
    ) else (
        start "" /b ollama serve >nul 2>&1
    )
    timeout /t 5 >nul
) else (
    echo Ollama service is already running.
)

echo.
echo ===================================================
echo Starting AgniAI Application...
echo ===================================================

if exist "%ROOT_DIR%agniai.exe" (
    start "" "%ROOT_DIR%agniai.exe"
) else if exist "%ROOT_DIR%dist\agniai\agniai.exe" (
    start "" "%ROOT_DIR%dist\agniai\agniai.exe"
) else if exist "%ROOT_DIR%_internal" (
    if exist "%ROOT_DIR%agniai" (
        start "" "%ROOT_DIR%agniai"
    )
) else (
    echo Running python app.py...
    python "%ROOT_DIR%app.py"
)

echo.
echo AgniAI launched successfully!
echo Open http://localhost:5000/api/health in your browser.
echo.
pause