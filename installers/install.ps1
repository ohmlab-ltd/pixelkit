# PixelKit dev/CLI install (Windows). Requires Python 3.11+ and Node.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.11+ required" }

Write-Host "-> engine venv"
python -m venv engine\.venv
$pip = "engine\.venv\Scripts\pip.exe"
& $pip install --upgrade pip | Out-Null

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  Write-Host "-> NVIDIA GPU detected - CUDA torch"
  & $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
}
& $pip install -e engine

if (Get-Command npm -ErrorAction SilentlyContinue) {
  Write-Host "-> building UI"
  Push-Location ui; npm ci; npm run build; Pop-Location
} else {
  Write-Host "! npm not found - engine API works, UI won't be served"
}
& "engine\.venv\Scripts\pixelkit.exe" doctor
Write-Host "Run: engine\.venv\Scripts\pixelkit.exe  -> http://127.0.0.1:8001"
