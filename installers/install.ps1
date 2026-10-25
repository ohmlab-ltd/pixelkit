# Sets up the engine and UI from a checkout, on Windows.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.11+ required" }

Write-Host "Creating engine\.venv"
python -m venv engine\.venv
$pip = "engine\.venv\Scripts\pip.exe"
& $pip install --upgrade pip | Out-Null

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  Write-Host "NVIDIA GPU found: installing CUDA PyTorch"
  & $pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
} else {
  Write-Host "No NVIDIA GPU: using the default PyTorch (CPU)"
}
& $pip install -e engine

if (Get-Command npm -ErrorAction SilentlyContinue) {
  Write-Host "Building the UI"
  Push-Location ui; npm ci; npm run build; Pop-Location
} else {
  Write-Host "npm not found. The engine API works, but the UI will not be served. Install Node, then run: cd ui; npm ci; npm run build"
}

Write-Host ""
& "engine\.venv\Scripts\pixelkit.exe" doctor
Write-Host "Run engine\.venv\Scripts\pixelkit.exe, then open http://127.0.0.1:8001"
