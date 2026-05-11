@echo off
setlocal

cd /d "%~dp0"
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
conda run -n physis_seg streamlit run scripts/30_streamlit_app.py --server.headless true --browser.gatherUsageStats false

endlocal
