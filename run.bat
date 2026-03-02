@echo off
cd /d "%~dp0"

if not exist "data" mkdir data
if not exist "data\chroma" mkdir data\chroma

pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -e .
)

echo.
echo Starting Chotgor at http://localhost:8000
echo UI: http://localhost:8000/ui/
echo Press Ctrl+C to stop.
echo.
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
