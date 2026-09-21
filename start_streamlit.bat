@echo off
setlocal

cd /d "%~dp0"
title CPAK Streamlit Launcher
set "STREAMLIT_SUPPRESS_EMAIL_PROMPT=true"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
set "STREAMLIT_SERVER_HEADLESS=true"
set "HOME=%USERPROFILE%"

if not exist "%USERPROFILE%\.streamlit" mkdir "%USERPROFILE%\.streamlit"
(
  echo [general]
  echo email = ""
) > "%USERPROFILE%\.streamlit\credentials.toml"
(
  echo [browser]
  echo gatherUsageStats = false
  echo serverAddress = "localhost"
) > "%USERPROFILE%\.streamlit\config.toml"

echo Abriendo navegador en http://localhost:8501 ...
start "" "http://localhost:8501"

echo Iniciando Streamlit CPAK...

set "CONDA_BAT="
if exist "%USERPROFILE%\Miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\Miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\Miniconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\Miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\Anaconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\Anaconda3\condabin\conda.bat"

if defined CONDA_BAT (
  call "%CONDA_BAT%" run -n physis_seg streamlit run scripts/30_streamlit_app.py --server.headless true --browser.gatherUsageStats false
) else (
  call conda run -n physis_seg streamlit run scripts/30_streamlit_app.py --server.headless true --browser.gatherUsageStats false
)

if errorlevel 1 (
  echo.
  echo ERROR: Streamlit no pudo iniciar. Revisa el mensaje anterior.
  echo Esta ventana se mantendra abierta para diagnostico.
  pause
  endlocal
  exit /b 1
)

endlocal
