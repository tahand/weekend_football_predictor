@echo off
rem Start Weekend Football Predictor, creating the virtual environment on first run.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    echo Creating virtual environment...
    py -3.13 -m venv .venv || goto :error
    .venv\Scripts\python -m pip install -r requirements.txt || goto :error
)
.venv\Scripts\python main.py
goto :eof
:error
echo Setup failed. Install Python 3.13 from python.org and try again.
pause
