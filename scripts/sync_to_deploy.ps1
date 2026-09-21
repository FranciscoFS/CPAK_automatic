# ---------------------------------------------------------------------------
# Sincroniza la app del repo de investigación al repo de despliegue y publica.
#
# Uso:
#     powershell -File scripts\sync_to_deploy.ps1
#     powershell -File scripts\sync_to_deploy.ps1 -Message "Mejora del overlay"
#
# Qué hace:
#   1. Copia los archivos de la app a deploy_app\scripts\
#   2. Verifica que las copias sean idénticas (hash SHA-256)
#   3. Commit + push en deploy_app  ->  Streamlit Cloud se actualiza solo
# ---------------------------------------------------------------------------
param(
    [string]$Message = "Sync app from research repo"
)

$ErrorActionPreference = "Stop"

$root   = Split-Path -Parent $PSScriptRoot
$srcDir = Join-Path $root "scripts"
$deploy = Join-Path $root "deploy_app"

if (-not (Test-Path $deploy)) {
    Write-Host "ERROR: no existe la carpeta del repo de despliegue: $deploy" -ForegroundColor Red
    exit 1
}

# Archivos que forman parte de la app web
$files = @(
    "30_streamlit_app.py",
    "13_final_inference.py"
)

Write-Host "Sincronizando archivos de la app..." -ForegroundColor Cyan
$failed = $false
foreach ($f in $files) {
    $from = Join-Path $srcDir $f
    $to   = Join-Path $deploy "scripts\$f"

    if (-not (Test-Path $from)) {
        Write-Host ("  {0,-26} NO EXISTE EN EL ORIGEN" -f $f) -ForegroundColor Red
        $failed = $true
        continue
    }

    Copy-Item $from $to -Force

    $ok = (Get-FileHash $from -Algorithm SHA256).Hash -eq (Get-FileHash $to -Algorithm SHA256).Hash
    Write-Host ("  {0,-26} {1}" -f $f, $(if ($ok) { "OK" } else { "ERROR DE COPIA" })) `
        -ForegroundColor $(if ($ok) { "Green" } else { "Red" })
    if (-not $ok) { $failed = $true }
}

if ($failed) {
    Write-Host "`nAbortado: hubo errores al copiar." -ForegroundColor Red
    exit 1
}

Push-Location $deploy
try {
    $pending = git status --porcelain
    if (-not $pending) {
        Write-Host "`nNo hay cambios que publicar (el repo de despliegue ya esta al dia)." -ForegroundColor Yellow
        return
    }

    Write-Host "`nCambios detectados:" -ForegroundColor Cyan
    git status --short

    git add -A
    git commit -m $Message | Out-Null
    git push origin main

    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nPublicado. Streamlit Cloud se actualizara en segundos." -ForegroundColor Green
        Write-Host "Si tocaste requirements.txt o packages.txt, hara un redeploy completo (varios minutos)." -ForegroundColor Yellow
    } else {
        Write-Host "`nEl push fallo. Revisa el mensaje anterior." -ForegroundColor Red
        exit 1
    }
}
finally {
    Pop-Location
}
