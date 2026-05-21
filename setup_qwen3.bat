@echo off
cd /d "%~dp0"

echo ----------------------------------------------------
echo Qwen3-Swallow-8B Model Setup for Ollama (Docker)
echo ----------------------------------------------------

:: Check if Ollama container is running
docker compose ps | findstr /i "ollama" >nul
if errorlevel 1 (
    echo [ERROR] Ollama container is not running.
    echo Please start it first using: docker compose up -d
    pause
    exit /b
)

:: Find the GGUF file in ollama\models
set "MODEL_DIR=ollama\models"
set "GGUF_FILE="
for %%F in ("%MODEL_DIR%\*.gguf") do (
    set "GGUF_FILE=%%~nxF"
    goto :found
)

echo [ERROR] No .gguf file found in %MODEL_DIR%.
echo Please make sure the download is complete.
pause
exit /b

:found
echo Found model file: %GGUF_FILE%
echo.

:: Create Modelfile
echo Generating Modelfile...
echo FROM /root/.ollama/models/%GGUF_FILE% > ollama\Modelfile
:: VRAM上限のコントロール（コンテキストサイズを4096に固定。長すぎる会話履歴でのVRAM溢れを防ぎます）
echo PARAMETER num_ctx 4096 >> ollama\Modelfile
:: AIが途中で切れないように最大出力トークンも増やしておきます
echo PARAMETER num_predict 1024 >> ollama\Modelfile

echo Modelfile has been created at .\ollama\Modelfile.
echo.
echo Building model in Ollama... (This might take a moment)
docker compose exec ollama ollama create qwen3-swallow -f /root/.ollama/Modelfile

echo.
echo ----------------------------------------------------
echo Setup Complete!
echo You can test the model by running the following command:
echo docker compose exec ollama ollama run qwen3-swallow
echo ----------------------------------------------------
pause
