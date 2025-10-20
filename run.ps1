param(
  [switch]$NoVenv
)

Write-Host "House Maxify — starting in development mode" -ForegroundColor Cyan

if (-not $NoVenv) {
  if (-not (Test-Path .venv)) {
    Write-Host "Creating virtual environment (.venv)..."
    py -m venv .venv
  }
  Write-Host "Activating virtual environment..."
  .\.venv\Scripts\Activate.ps1
}

Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Launching Flask app on http://localhost:5000 ..." -ForegroundColor Green
python app.py

