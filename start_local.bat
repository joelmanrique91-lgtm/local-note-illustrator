@echo off
setlocal

cd /d %~dp0

echo [INFO] Running environment check...
python scripts\check_env.py
if errorlevel 1 (
  echo [ERROR] check_env failed. Fix environment before launching.
  exit /b 1
)

echo [INFO] Launching GUI...
python run_app.py
