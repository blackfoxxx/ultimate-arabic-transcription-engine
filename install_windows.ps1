# Ultimate Arabic Transcription Engine - Windows 11 Silent Installation Script
# This script installs all required dependencies and sets up the platform automatically
# No user interaction required - fully automated installation

param(
    [switch]$SkipPython,
    [switch]$SkipFFmpeg,
    [switch]$SkipGit,
    [string]$InstallPath = "C:\ArabicSTT"
)

# Set execution policy for this session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# Enable TLS 1.2 for secure downloads
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072

Write-Host "🎙️ Ultimate Arabic Transcription Engine - Windows 11 Silent Installation" -ForegroundColor Cyan
Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check for administrator privileges
if (-not (Test-Administrator)) {
    Write-Host "❌ This script requires administrator privileges" -ForegroundColor Red
    Write-Host "💡 Please run PowerShell as Administrator and try again" -ForegroundColor Yellow
    exit 1
}

# Function to download file with progress
function Download-File {
    param(
        [string]$Url,
        [string]$OutputPath
    )
    
    try {
        Write-Host "📥 Downloading: $([System.IO.Path]::GetFileName($OutputPath))" -ForegroundColor Yellow
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($Url, $OutputPath)
        Write-Host "✅ Downloaded successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Download failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install Chocolatey
function Install-Chocolatey {
    if (Get-Command choco -ErrorAction SilentlyContinue) {
        Write-Host "✅ Chocolatey is already installed" -ForegroundColor Green
        return $true
    }
    
    Write-Host "📦 Installing Chocolatey package manager..." -ForegroundColor Yellow
    try {
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        Write-Host "✅ Chocolatey installed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to install Chocolatey: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install Python
function Install-Python {
    if ($SkipPython) {
        Write-Host "⏭️ Skipping Python installation (--SkipPython flag)" -ForegroundColor Yellow
        return $true
    }
    
    # Check if Python 3.8+ is already installed
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match "Python 3\.([8-9]|\d{2,})") {
            Write-Host "✅ Python 3.8+ is already installed: $pythonVersion" -ForegroundColor Green
            return $true
        }
    }
    catch {
        # Python not found, continue with installation
    }
    
    Write-Host "🐍 Installing Python 3.11..." -ForegroundColor Yellow
    try {
        choco install python311 -y --force
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        Write-Host "✅ Python 3.11 installed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to install Python: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install FFmpeg
function Install-FFmpeg {
    if ($SkipFFmpeg) {
        Write-Host "⏭️ Skipping FFmpeg installation (--SkipFFmpeg flag)" -ForegroundColor Yellow
        return $true
    }
    
    if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
        Write-Host "✅ FFmpeg is already installed" -ForegroundColor Green
        return $true
    }
    
    Write-Host "🎬 Installing FFmpeg..." -ForegroundColor Yellow
    try {
        choco install ffmpeg -y --force
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        Write-Host "✅ FFmpeg installed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to install FFmpeg: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install Git
function Install-Git {
    if ($SkipGit) {
        Write-Host "⏭️ Skipping Git installation (--SkipGit flag)" -ForegroundColor Yellow
        return $true
    }
    
    if (Get-Command git -ErrorAction SilentlyContinue) {
        Write-Host "✅ Git is already installed" -ForegroundColor Green
        return $true
    }
    
    Write-Host "📚 Installing Git..." -ForegroundColor Yellow
    try {
        choco install git -y --force
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        Write-Host "✅ Git installed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to install Git: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install Visual C++ Redistributables
function Install-VCRedist {
    Write-Host "🔧 Installing Visual C++ Redistributables..." -ForegroundColor Yellow
    try {
        choco install vcredist-all -y --force
        Write-Host "✅ Visual C++ Redistributables installed" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to install Visual C++ Redistributables: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to create installation directory
function Create-InstallDirectory {
    Write-Host "📁 Creating installation directory: $InstallPath" -ForegroundColor Yellow
    try {
        if (-not (Test-Path $InstallPath)) {
            New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
        }
        Set-Location $InstallPath
        Write-Host "✅ Installation directory created" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to create installation directory: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to clone or copy the repository
function Setup-Repository {
    Write-Host "📦 Setting up Arabic Transcription Engine..." -ForegroundColor Yellow
    
    # If we're already in the project directory, just continue
    if (Test-Path "requirements.txt") {
        Write-Host "✅ Already in project directory" -ForegroundColor Green
        return $true
    }
    
    # Try to copy from current directory if it exists
    $currentDir = Get-Location
    $sourceDir = Split-Path -Parent $PSScriptRoot
    
    if (Test-Path "$sourceDir\requirements.txt") {
        Write-Host "📋 Copying project files from current directory..." -ForegroundColor Yellow
        try {
            Copy-Item -Path "$sourceDir\*" -Destination $InstallPath -Recurse -Force
            Write-Host "✅ Project files copied successfully" -ForegroundColor Green
            return $true
        }
        catch {
            Write-Host "❌ Failed to copy project files: $($_.Exception.Message)" -ForegroundColor Red
            return $false
        }
    }
    
    Write-Host "❌ Could not find project files. Please ensure the script is run from the project directory." -ForegroundColor Red
    return $false
}

# Function to create Python virtual environment
function Create-VirtualEnvironment {
    Write-Host "🌐 Creating Python virtual environment..." -ForegroundColor Yellow
    try {
        python -m venv venv
        
        # Activate virtual environment
        & ".\venv\Scripts\Activate.ps1"
        
        # Upgrade pip
        Write-Host "📦 Upgrading pip..." -ForegroundColor Yellow
        python -m pip install --upgrade pip --quiet
        
        Write-Host "✅ Virtual environment created and activated" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to create virtual environment: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install Python dependencies
function Install-PythonDependencies {
    Write-Host "📚 Installing Python dependencies..." -ForegroundColor Yellow
    try {
        # Install PyTorch first (CPU version for compatibility)
        Write-Host "🔥 Installing PyTorch..." -ForegroundColor Yellow
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
        
        # Install other requirements
        Write-Host "📋 Installing other requirements..." -ForegroundColor Yellow
        pip install -r requirements.txt --quiet
        
        Write-Host "✅ Python dependencies installed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to install Python dependencies: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "💡 Try running: pip install --upgrade pip" -ForegroundColor Yellow
        return $false
    }
}

# Function to create environment configuration
function Create-EnvironmentConfig {
    Write-Host "⚙️ Setting up environment configuration..." -ForegroundColor Yellow
    
    if (Test-Path ".env") {
        Write-Host "ℹ️ Environment configuration already exists" -ForegroundColor Blue
        return $true
    }
    
    try {
        $secretKey = -join ((1..64) | ForEach {Get-Random -input ([char[]]([char]'a'..[char]'z') + ([char[]]([char]'A'..[char]'Z')) + 0..9)})
        
        $envContent = @"
# Ultimate Arabic Transcription Engine Configuration
# Processing Mode: 'local' for Whisper models, 'api' for OpenAI API
PROCESSING_MODE=local

# OpenAI API Configuration (optional)
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=

# Whisper Model Settings
WHISPER_MODEL_SIZE=medium
WHISPER_DEVICE=auto

# Server Settings
HOST=0.0.0.0
PORT=5002
DEBUG=False

# Security
SECRET_KEY=$secretKey
API_KEY_REQUIRED=False

# Logging
LOG_LEVEL=INFO

# Windows-specific settings
TEMP_DIR=temp
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
"@
        
        Set-Content -Path ".env" -Value $envContent -Encoding UTF8
        Write-Host "✅ Environment configuration created (.env)" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to create environment configuration: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to create required directories
function Create-RequiredDirectories {
    Write-Host "📁 Creating required directories..." -ForegroundColor Yellow
    try {
        $directories = @("temp", "uploads", "outputs", "models", "logs")
        foreach ($dir in $directories) {
            if (-not (Test-Path $dir)) {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
            }
        }
        Write-Host "✅ Required directories created" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to create directories: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to download initial Whisper model
function Download-WhisperModel {
    Write-Host "🤖 Downloading initial Whisper model..." -ForegroundColor Yellow
    try {
        python -c "
try:
    from faster_whisper import WhisperModel
    import os
    os.makedirs('models', exist_ok=True)
    model = WhisperModel('medium', download_root='./models')
    print('✅ Medium Whisper model downloaded successfully')
except Exception as e:
    print(f'⚠️ Failed to download model: {e}')
    print('Models will be downloaded on first use')
"
        return $true
    }
    catch {
        Write-Host "⚠️ Failed to download initial model - models will be downloaded on first use" -ForegroundColor Yellow
        return $true
    }
}

# Function to create Windows service script
function Create-ServiceScript {
    Write-Host "🔧 Creating Windows service script..." -ForegroundColor Yellow
    try {
        $serviceScript = @"
@echo off
cd /d "$InstallPath"
call venv\Scripts\activate.bat
python app.py
pause
"@
        
        Set-Content -Path "start_service.bat" -Value $serviceScript -Encoding ASCII
        
        $startScript = @"
@echo off
echo Starting Ultimate Arabic Transcription Engine...
cd /d "$InstallPath"
call venv\Scripts\activate.bat
start /min python app.py
echo Service started. Access at http://localhost:5002
timeout /t 3 /nobreak >nul
"@
        
        Set-Content -Path "start.bat" -Value $startScript -Encoding ASCII
        
        Write-Host "✅ Service scripts created" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "❌ Failed to create service scripts: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to create desktop shortcut
function Create-DesktopShortcut {
    Write-Host "🖥️ Creating desktop shortcut..." -ForegroundColor Yellow
    try {
        $WshShell = New-Object -comObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\Arabic Transcription Engine.lnk")
        $Shortcut.TargetPath = "$InstallPath\start.bat"
        $Shortcut.WorkingDirectory = $InstallPath
        $Shortcut.Description = "Ultimate Arabic Transcription Engine"
        $Shortcut.Save()
        
        Write-Host "✅ Desktop shortcut created" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "⚠️ Failed to create desktop shortcut: $($_.Exception.Message)" -ForegroundColor Yellow
        return $true
    }
}

# Function to test installation
function Test-Installation {
    Write-Host "🧪 Testing installation..." -ForegroundColor Yellow
    try {
        # Test Python imports
        python -c "
import sys
print(f'Python version: {sys.version}')

# Test core imports
try:
    import torch
    print('✅ PyTorch imported successfully')
except ImportError as e:
    print(f'❌ PyTorch import failed: {e}')

try:
    import whisper
    print('✅ Whisper imported successfully')
except ImportError as e:
    print(f'❌ Whisper import failed: {e}')

try:
    from faster_whisper import WhisperModel
    print('✅ Faster-Whisper imported successfully')
except ImportError as e:
    print(f'❌ Faster-Whisper import failed: {e}')

try:
    import flask
    print('✅ Flask imported successfully')
except ImportError as e:
    print(f'❌ Flask import failed: {e}')
"
        Write-Host "✅ Installation test completed" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "⚠️ Installation test encountered issues: $($_.Exception.Message)" -ForegroundColor Yellow
        return $true
    }
}

# Main installation process
Write-Host "🚀 Starting silent installation process..." -ForegroundColor Cyan
Write-Host ""

$installSteps = @(
    @{ Name = "Installing Chocolatey"; Function = { Install-Chocolatey } },
    @{ Name = "Installing Python 3.11"; Function = { Install-Python } },
    @{ Name = "Installing FFmpeg"; Function = { Install-FFmpeg } },
    @{ Name = "Installing Git"; Function = { Install-Git } },
    @{ Name = "Installing Visual C++ Redistributables"; Function = { Install-VCRedist } },
    @{ Name = "Creating installation directory"; Function = { Create-InstallDirectory } },
    @{ Name = "Setting up repository"; Function = { Setup-Repository } },
    @{ Name = "Creating virtual environment"; Function = { Create-VirtualEnvironment } },
    @{ Name = "Installing Python dependencies"; Function = { Install-PythonDependencies } },
    @{ Name = "Creating environment configuration"; Function = { Create-EnvironmentConfig } },
    @{ Name = "Creating required directories"; Function = { Create-RequiredDirectories } },
    @{ Name = "Downloading Whisper model"; Function = { Download-WhisperModel } },
    @{ Name = "Creating service scripts"; Function = { Create-ServiceScript } },
    @{ Name = "Creating desktop shortcut"; Function = { Create-DesktopShortcut } },
    @{ Name = "Testing installation"; Function = { Test-Installation } }
)

$successCount = 0
$totalSteps = $installSteps.Count

foreach ($step in $installSteps) {
    Write-Host "[$($successCount + 1)/$totalSteps] $($step.Name)..." -ForegroundColor Cyan
    if (& $step.Function) {
        $successCount++
    } else {
        Write-Host "❌ Installation step failed: $($step.Name)" -ForegroundColor Red
        Write-Host "💡 You may need to run this step manually" -ForegroundColor Yellow
    }
    Write-Host ""
}

# Installation summary
Write-Host "🎉 Installation Summary" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host "✅ Completed steps: $successCount/$totalSteps" -ForegroundColor Green
Write-Host "📍 Installation path: $InstallPath" -ForegroundColor Blue
Write-Host ""

if ($successCount -eq $totalSteps) {
    Write-Host "🎉 Installation completed successfully!" -ForegroundColor Green
} else {
    Write-Host "⚠️ Installation completed with some issues" -ForegroundColor Yellow
    Write-Host "💡 Check the output above for any failed steps" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🚀 Quick Start:" -ForegroundColor Cyan
Write-Host "1. Double-click 'Arabic Transcription Engine' on your desktop" -ForegroundColor White
Write-Host "   OR" -ForegroundColor Yellow
Write-Host "2. Run: $InstallPath\start.bat" -ForegroundColor White
Write-Host "3. Open your browser to: http://localhost:5002" -ForegroundColor White
Write-Host ""
Write-Host "⚙️ Configuration:" -ForegroundColor Cyan
Write-Host "- Edit $InstallPath\.env to configure settings" -ForegroundColor White
Write-Host "- Visit http://localhost:5002/settings for web configuration" -ForegroundColor White
Write-Host ""
Write-Host "📖 Documentation:" -ForegroundColor Cyan
Write-Host "- README.md - Complete setup guide" -ForegroundColor White
Write-Host "- Check $InstallPath for all project files" -ForegroundColor White
Write-Host ""

# Pause to show results
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")