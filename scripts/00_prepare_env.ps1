param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath ".venv")) {
  & $PythonExe -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install -e .

Write-Host "Environment ready. Activate with: .\\.venv\\Scripts\\Activate.ps1"
