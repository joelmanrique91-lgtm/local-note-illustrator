@echo off
setlocal

cd /d %~dp0

echo [INFO] Local Note Illustrator - Inicio local
echo [INFO] Tip: ejecuta este script desde tu entorno virtual/conda ya activado.
echo [INFO] Si faltan dependencias, usa: pip install -r requirements.txt

echo [INFO] Running environment check...
python scripts\check_env.py
if errorlevel 1 (
  echo [ERROR] check_env failed. Revisa Python/dependencias/config y vuelve a intentar.
  exit /b 1
)

echo [INFO] Environment check OK. Launching GUI...
python run_app.py
