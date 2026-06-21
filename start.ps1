#Build C++ backend and start Node.js server with ML features.

Write-Host "Building C++ Backend..." -ForegroundColor Cyan
if (-not (Test-Path "build")) { New-Item -ItemType Directory -Force build }

Set-Location build
cmake -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ..
cmake --build . --config Debug

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "Build successful! Binary located at build/Debug/dsp.exe" -ForegroundColor Green
Write-Host "Verification tool at build/Debug/verify_cpp.exe" -ForegroundColor Green

Write-Host "`n" -ForegroundColor Cyan
Write-Host "ML-Based Audio Engine Startup" -ForegroundColor Magenta

# Check if features are extracted
if (-not (Test-Path "../tools/extracted_features")) {
    Write-Host "`nWarning: ML Features not extracted!" -ForegroundColor Yellow
    Write-Host "Run: python tools/extract_features_real.py" -ForegroundColor Yellow
    Write-Host "This extracts features from the FMA dataset." -ForegroundColor Yellow
}

# Install Node dependencies if needed
if (-not (Test-Path "../ui/node_modules")) {
    Write-Host "`nInstalling Node.js dependencies..." -ForegroundColor Cyan
    Set-Location ../ui
    npm install
    Set-Location ../build
}

Write-Host "`nStarting ML-Based Audio Engine" -ForegroundColor Green
Write-Host "Server will be available at: http://localhost:3000" -ForegroundColor Green
Write-Host "`nFeatures:" -ForegroundColor Cyan
Write-Host "ML Dynamic EQ (learns from real audio)" -ForegroundColor Green
Write-Host "Genre Detection (16 genres)" -ForegroundColor Green
Write-Host "Real-time DSP (5.8ms latency)" -ForegroundColor Green
Write-Host "MP3 + WAV Support" -ForegroundColor Green
Write-Host "`n"

Set-Location ../ui
npm start
