# ML Service Startup Script for PowerShell
# Starts the Flask ML service with proper environment setup

Write-Host "======================================================" -ForegroundColor Green
Write-Host "Starting ML Service (MEAI Framework)..." -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green

# Prefer venv inside ml_model if available, otherwise use system Python
$pythonExePath = Join-Path (Get-Location).Path ".venv\Scripts\python.exe"
if (Test-Path $pythonExePath) {
    Write-Host "[OK] Using virtual environment Python at: $pythonExePath" -ForegroundColor Green
} else {
    $pythonExePath = "python"
    Write-Host "[INFO] Local venv not found at .\\.venv\\Scripts\\python.exe - using system Python" -ForegroundColor Yellow
}

# Check if Python is available
try {
    $pythonVersion = & "$pythonExePath" --version 2>&1
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found. Please install Python 3.8+ or create .venv" -ForegroundColor Red
    exit 1
}

# Check if we're in the right directory
if (-not (Test-Path "service.py")) {
    Write-Host "[ERROR] service.py not found. Please run this script from the ml_model directory." -ForegroundColor Red
    exit 1
}

# Check dependencies
Write-Host "" 
Write-Host "Checking dependencies..." -ForegroundColor Yellow
& "$pythonExePath" -c "import flask, flask_cors, numpy, pandas, sklearn, requests"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Missing dependencies detected. Installing..." -ForegroundColor Yellow
    & "$pythonExePath" -m pip install --upgrade pip
    & "$pythonExePath" -m pip install -r requirements.txt
} else {
    Write-Host "[OK] Core ML dependencies available" -ForegroundColor Green
}

# Start the service
Write-Host ""
Write-Host "Starting ML Service..." -ForegroundColor Green
Write-Host "Service URL: http://127.0.0.1:5000 (Local Development)" -ForegroundColor Cyan
Write-Host "Production URL: https://endpoint-pmoc-a0a6708d039f.herokuapp.com" -ForegroundColor Cyan
Write-Host "MEAI Categories: 4 (Marriage, Parenthood, Family Planning, Health)" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the service" -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""

try {
    & "$pythonExePath" service.py
} catch {
    Write-Host ""
    Write-Host "[ERROR] Failed to start service: $_" -ForegroundColor Red
    Write-Host ("Try running manually: `"{0}`" service.py" -f $pythonExePath) -ForegroundColor Yellow
}
