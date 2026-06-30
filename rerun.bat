@echo off
REM Chotgor backend restart helper.
REM Find the process listening on port 8000, kill it, then relaunch run.bat with debug on.
REM NOTE: keep this file ASCII-only. cmd reads .bat as the system codepage (CP932 here),
REM so non-ASCII comments get mis-decoded and can break parsing (stray & etc).
cd /d "%~dp0"

for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Killing PID %%p on port 8000...
    taskkill /F /T /PID %%p >nul 2>&1
)

call "%~dp0run.bat" -debug on
