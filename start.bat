@echo off
echo Starting Ollama...
start "" "C:\Users\dhruv\AppData\Local\Programs\Ollama\ollama.exe" serve
echo Waiting for Ollama...
timeout /t 15
echo Starting AgniAI...
start "" "E:\AgniAI\dist\agniai\agniai.exe"
echo.
echo Done. Open http://localhost:5000/api/health
pause