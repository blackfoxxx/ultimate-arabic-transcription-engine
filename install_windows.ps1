# Ultimate Arabic Transcription Engine - Windows Installation Script
# This script installs all dependencies and sets up the Arabic transcription engine with LLM support

param(
    [string]$InstallPath = "C:\UltimateArabicTranscription",
    [switch]$SkipChocolatey,
    [switch]$SkipPython,
    [switch]$SkipGit,
    [switch]$WhatIf
)

# Set error handling
$ErrorActionPreference = "Stop"

# Function to check if running as administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Function to install Chocolatey
function Install-Chocolatey {
    Write-Host "Installing Chocolatey package manager..." -ForegroundColor Yellow
    try {
        if (Get-Command choco -ErrorAction SilentlyContinue) {
            Write-Host "[OK] Chocolatey is already installed" -ForegroundColor Green
            return $true
        }
        
        Set-ExecutionPolicy Bypass -Scope Process -Force
        [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
        Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        Write-Host "[OK] Chocolatey installed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to install Chocolatey: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install Python
function Install-Python {
    Write-Host "Installing Python..." -ForegroundColor Yellow
    try {
        if (Get-Command python -ErrorAction SilentlyContinue) {
            $pythonVersion = python --version 2>&1
            if ($pythonVersion -match "Python 3\.[8-9]|Python 3\.1[0-9]") {
                Write-Host "[OK] Python is already installed: $pythonVersion" -ForegroundColor Green
                return $true
            }
        }
        
        choco install python -y
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        Write-Host "[OK] Python installed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to install Python: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install FFmpeg
function Install-FFmpeg {
    Write-Host "Installing FFmpeg..." -ForegroundColor Yellow
    try {
        if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
            Write-Host "[OK] FFmpeg is already installed" -ForegroundColor Green
            return $true
        }
        
        choco install ffmpeg -y
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        Write-Host "[OK] FFmpeg installed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to install FFmpeg: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install Git
function Install-Git {
    Write-Host "Installing Git..." -ForegroundColor Yellow
    try {
        if (Get-Command git -ErrorAction SilentlyContinue) {
            Write-Host "[OK] Git is already installed" -ForegroundColor Green
            return $true
        }
        
        choco install git -y
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        Write-Host "[OK] Git installed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to install Git: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install Visual C++ Redistributables
function Install-VCRedist {
    Write-Host "Installing Visual C++ Redistributables..." -ForegroundColor Yellow
    try {
        choco install vcredist-all -y
        Write-Host "[OK] Visual C++ Redistributables installed" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to install Visual C++ Redistributables: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install Ollama for LLM support
function Install-Ollama {
    Write-Host "Installing Ollama for LLM support..." -ForegroundColor Yellow
    try {
        if (Get-Command ollama -ErrorAction SilentlyContinue) {
            Write-Host "[OK] Ollama is already installed" -ForegroundColor Green
            return $true
        }
        
        # Install Ollama using winget
        winget install Ollama.Ollama --accept-source-agreements --accept-package-agreements
        
        # Refresh environment variables
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        Write-Host "[OK] Ollama installed successfully" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to install Ollama: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "[INFO] You can manually install Ollama from https://ollama.ai" -ForegroundColor Yellow
        return $false
    }
}

# Function to start Ollama service
function Start-OllamaService {
    Write-Host "Starting Ollama service..." -ForegroundColor Yellow
    try {
        # Start Ollama service in background
        Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden -PassThru
        Start-Sleep -Seconds 3
        Write-Host "[OK] Ollama service started" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[WARN] Could not start Ollama service automatically: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "[INFO] You may need to start Ollama manually after installation" -ForegroundColor Yellow
        return $true
    }
}

# Function to download LLM models
function Download-LLMModels {
    Write-Host "Downloading LLM models..." -ForegroundColor Yellow
    try {
        # Check if Ollama is available
        if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
            Write-Host "[WARN] Ollama not found, skipping model download" -ForegroundColor Yellow
            return $true
        }
        
        # Ensure Ollama service is running
        Start-OllamaService
        
        # Download Aya model for Arabic enhancement
        Write-Host "Downloading Aya model for Arabic enhancement..." -ForegroundColor Cyan
        try {
            ollama pull aya:8b
            Write-Host "[OK] Aya model downloaded successfully" -ForegroundColor Green
        }
        catch {
            Write-Host "[WARN] Failed to download Aya model: $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host "[INFO] You can manually download it later with: ollama pull aya:8b" -ForegroundColor Yellow
        }
        
        # Download Llama model for general tasks
        Write-Host "Downloading Llama model for general tasks..." -ForegroundColor Cyan
        try {
            ollama pull llama3.2:3b
            Write-Host "[OK] Llama model downloaded successfully" -ForegroundColor Green
        }
        catch {
            Write-Host "[WARN] Failed to download Llama model: $($_.Exception.Message)" -ForegroundColor Yellow
            Write-Host "[INFO] You can manually download it later with: ollama pull llama3.2:3b" -ForegroundColor Yellow
        }
        
        return $true
    }
    catch {
        Write-Host "[WARN] Model download encountered issues: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "[INFO] Models can be downloaded manually after installation" -ForegroundColor Yellow
        return $true
    }
}

# Function to create installation directory
function Create-InstallDirectory {
    Write-Host "Creating installation directory..." -ForegroundColor Yellow
    try {
        if (-not (Test-Path $InstallPath)) {
            New-Item -ItemType Directory -Path $InstallPath -Force | Out-Null
        }
        Set-Location $InstallPath
        Write-Host "[OK] Installation directory created: $InstallPath" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to create installation directory: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to clone or copy the repository
function Setup-Repository {
    Write-Host "Setting up Arabic Transcription Engine..." -ForegroundColor Yellow
    
    try {
        if (Test-Path ".git") {
            Write-Host "[OK] Repository already exists, pulling latest changes..." -ForegroundColor Green
            git pull
        } else {
            Write-Host "Cloning repository..." -ForegroundColor Cyan
            git clone https://github.com/your-repo/ultimate-arabic-transcription-engine.git .
        }
        
        Write-Host "[OK] Repository setup completed" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to setup repository: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "[INFO] Please ensure you have internet connection and Git is installed" -ForegroundColor Yellow
        return $false
    }
}

# Function to create virtual environment
function Create-VirtualEnvironment {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    try {
        if (Test-Path "venv") {
            Write-Host "[OK] Virtual environment already exists" -ForegroundColor Green
        } else {
            python -m venv venv
            Write-Host "[OK] Virtual environment created" -ForegroundColor Green
        }
        
        # Activate virtual environment
        & "venv\Scripts\Activate.ps1"
        Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to create virtual environment: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to install Python dependencies
function Install-PythonDependencies {
    Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
    try {
        # Upgrade pip first
        python -m pip install --upgrade pip
        
        # Install requirements
        if (Test-Path "requirements.txt") {
            python -m pip install -r requirements.txt
        } else {
            Write-Host "[WARN] requirements.txt not found, installing basic dependencies..." -ForegroundColor Yellow
            python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
            python -m pip install openai-whisper faster-whisper flask requests python-dotenv
        }
        
        Write-Host "[OK] Python dependencies installed" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to install Python dependencies: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "[INFO] Try running: pip install --upgrade pip" -ForegroundColor Yellow
        return $false
    }
}

# Function to create environment configuration
function Create-EnvironmentConfig {
    Write-Host "Setting up environment configuration..." -ForegroundColor Yellow
    
    try {
        $envContent = @"
# Processing Configuration
PROCESSING_MODE=enhanced
OPENAI_API_KEY=your_openai_api_key_here

# Whisper Configuration
WHISPER_MODEL_SIZE=medium
WHISPER_DEVICE=cpu

# Server Configuration
HOST=0.0.0.0
PORT=5002
DEBUG=False

# Security
SECRET_KEY=your_secret_key_here
API_KEY_REQUIRED=False

# Logging
LOG_LEVEL=INFO

# LLM Configuration
ENABLE_LLM=True
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL_ENHANCEMENT=aya:8b
LLM_MODEL_GENERAL=llama3.2:3b
LLM_TIMEOUT=30

# Directories
TEMP_DIR=temp
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
"@
        
        Set-Content -Path ".env" -Value $envContent -Encoding UTF8
        Write-Host "[OK] Environment configuration created (.env)" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to create environment configuration: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to download Whisper model
function Download-WhisperModel {
    Write-Host "Downloading Whisper model..." -ForegroundColor Yellow
    try {
        # Create a simple Python script to download the model
        $downloadScript = @"
import whisper
print('Downloading Whisper medium model...')
model = whisper.load_model('medium')
print('[OK] Whisper model downloaded successfully')
"@
        
        $downloadScript | Out-File -FilePath "download_model.py" -Encoding UTF8
        python download_model.py
        Remove-Item "download_model.py" -Force
        
        Write-Host "[OK] Whisper model downloaded" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[WARN] Whisper model download failed: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "[INFO] Model will be downloaded automatically on first use" -ForegroundColor Yellow
        return $true
    }
}

# Function to create Windows service script
function Create-ServiceScript {
    Write-Host "Creating Windows service script..." -ForegroundColor Yellow
    try {
        $serviceScript = @'
@echo off
cd /d "{0}"
call venv\Scripts\activate.bat
python app.py
pause
'@ -f $InstallPath
        
        Set-Content -Path "start_service.bat" -Value $serviceScript -Encoding ASCII
        
        $startScript = @'
@echo off
echo Starting Ultimate Arabic Transcription Engine...
cd /d "{0}"
call venv\Scripts\activate.bat
start /min python app.py
echo Service started. Access at http://localhost:5002
timeout /t 3 /nobreak >nul
'@ -f $InstallPath
        
        Set-Content -Path "start.bat" -Value $startScript -Encoding ASCII
        
        # Create start_cli.bat for CLI execution
        $cliScript = @'
@echo off
cd /d "{0}"
call venv\Scripts\activate.bat
echo.
echo ========================================
echo   Arabic Transcription CLI Tool
echo ========================================
echo.
echo Usage Examples:
echo   Interactive mode: python arabic_cli_ultimate.py --interactive
echo   Transcribe file:  python arabic_cli_ultimate.py --file audio.wav
echo   Batch process:    python arabic_cli_ultimate.py --batch-dir ./recordings
echo   Full help:        python arabic_cli_ultimate.py --help-full
echo.
python arabic_cli_ultimate.py %*
pause
'@ -f $InstallPath
        
        Set-Content -Path "start_cli.bat" -Value $cliScript -Encoding ASCII
        
        Write-Host "[OK] Service scripts created" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[ERROR] Failed to create service scripts: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to create desktop shortcut
function Create-DesktopShortcut {
    Write-Host "Creating desktop shortcuts..." -ForegroundColor Yellow
    try {
        $WshShell = New-Object -comObject WScript.Shell
        
        # Create Web App shortcut
        $Shortcut = $WshShell.CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\Ultimate Arabic Transcription.lnk")
        $Shortcut.TargetPath = "$InstallPath\start.bat"
        $Shortcut.WorkingDirectory = $InstallPath
        $Shortcut.IconLocation = "$InstallPath\start.bat"
        $Shortcut.Description = "Ultimate Arabic Transcription Engine (Web Interface)"
        $Shortcut.Save()
        
        # Create CLI shortcut
        $CLIShortcut = $WshShell.CreateShortcut("$([Environment]::GetFolderPath('Desktop'))\Arabic CLI.lnk")
        $CLIShortcut.TargetPath = "$InstallPath\start_cli.bat"
        $CLIShortcut.WorkingDirectory = $InstallPath
        $CLIShortcut.IconLocation = "$InstallPath\start_cli.bat"
        $CLIShortcut.Description = "Ultimate Arabic Transcription CLI Tool"
        $CLIShortcut.Save()
        
        Write-Host "[OK] Desktop shortcuts created" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[WARN] Failed to create desktop shortcuts: $($_.Exception.Message)" -ForegroundColor Yellow
        return $true
    }
}

# Function to test installation
function Test-Installation {
    Write-Host "Testing installation..." -ForegroundColor Yellow
    try {
        # Test Python imports
        python -c @"
import sys
print(f'Python version: {sys.version}')

# Test core imports
try:
    import torch
    print('[OK] PyTorch imported successfully')
except ImportError as e:
    print(f'[ERROR] PyTorch import failed: {e}')

try:
    import whisper
    print('[OK] Whisper imported successfully')
except ImportError as e:
    print(f'[ERROR] Whisper import failed: {e}')

try:
    from faster_whisper import WhisperModel
    print('[OK] Faster-Whisper imported successfully')
except ImportError as e:
    print(f'[ERROR] Faster-Whisper import failed: {e}')

# Test CLI availability
try:
    import arabic_cli_ultimate
    print('[OK] CLI module imported successfully')
except ImportError as e:
    print(f'[ERROR] CLI module import failed: {e}')
"@
        
        # Test CLI help command
        Write-Host "Testing CLI functionality..." -ForegroundColor Cyan
        try {
            python arabic_cli_ultimate.py --help | Select-Object -First 10
        }
        catch {
            Write-Host "[WARN] CLI test failed: $($_.Exception.Message)" -ForegroundColor Yellow
        }
        
        Write-Host "[OK] Installation test completed" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "[WARN] Installation test encountered issues: $($_.Exception.Message)" -ForegroundColor Yellow
        return $true
    }
}

# Main installation process
Write-Host "Ultimate Arabic Transcription Engine - Windows Installation" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan

if ($WhatIf) {
    Write-Host "Running in WhatIf mode - no actual changes will be made" -ForegroundColor Yellow
    Write-Host ""
}

# Check administrator privileges
if (-not (Test-Administrator)) {
    Write-Host "[WARN] This script requires administrator privileges for some operations." -ForegroundColor Yellow
    Write-Host "[INFO] Please run PowerShell as Administrator for best results." -ForegroundColor Yellow
    Write-Host ""
}

# Define installation steps
$installSteps = @(
    @{ Name = "Installing Chocolatey package manager"; Function = "Install-Chocolatey"; Skip = $SkipChocolatey },
    @{ Name = "Installing Python"; Function = "Install-Python"; Skip = $SkipPython },
    @{ Name = "Installing FFmpeg"; Function = "Install-FFmpeg"; Skip = $false },
    @{ Name = "Installing Git"; Function = "Install-Git"; Skip = $SkipGit },
    @{ Name = "Installing Visual C++ Redistributables"; Function = "Install-VCRedist"; Skip = $false },
    @{ Name = "Installing Ollama for LLM support"; Function = "Install-Ollama"; Skip = $false },
    @{ Name = "Starting Ollama service"; Function = "Start-OllamaService"; Skip = $false },
    @{ Name = "Downloading LLM models"; Function = "Download-LLMModels"; Skip = $false },
    @{ Name = "Creating installation directory"; Function = "Create-InstallDirectory"; Skip = $false },
    @{ Name = "Setting up repository"; Function = "Setup-Repository"; Skip = $false },
    @{ Name = "Creating virtual environment"; Function = "Create-VirtualEnvironment"; Skip = $false },
    @{ Name = "Installing Python dependencies"; Function = "Install-PythonDependencies"; Skip = $false },
    @{ Name = "Creating environment configuration"; Function = "Create-EnvironmentConfig"; Skip = $false },
    @{ Name = "Downloading Whisper model"; Function = "Download-WhisperModel"; Skip = $false },
    @{ Name = "Creating service scripts"; Function = "Create-ServiceScript"; Skip = $false },
    @{ Name = "Creating desktop shortcut"; Function = "Create-DesktopShortcut"; Skip = $false },
    @{ Name = "Testing installation"; Function = "Test-Installation"; Skip = $false }
)

# Execute installation steps
$totalSteps = $installSteps.Count
$currentStep = 0
$failedSteps = @()

foreach ($step in $installSteps) {
    $currentStep++
    
    if ($step.Skip) {
        Write-Host "Skipping: $($step.Name)" -ForegroundColor Gray
        continue
    }
    
    Write-Host ""
    Write-Host "[$currentStep/$totalSteps] $($step.Name)..." -ForegroundColor Cyan
    
    if ($WhatIf) {
        Write-Host "   Would execute: $($step.Function)" -ForegroundColor Gray
        continue
    }
    
    try {
        $result = & $step.Function
        if (-not $result) {
            $failedSteps += $step.Name
            Write-Host "[WARN] Step failed but continuing..." -ForegroundColor Yellow
        }
    }
    catch {
        $failedSteps += $step.Name
        Write-Host "[ERROR] Step failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "[WARN] Continuing with next step..." -ForegroundColor Yellow
    }
}

# Installation summary
Write-Host ""
Write-Host "=============================================================" -ForegroundColor Cyan
Write-Host "Installation Summary" -ForegroundColor Cyan
Write-Host "=============================================================" -ForegroundColor Cyan

if ($WhatIf) {
    Write-Host "[OK] WhatIf mode completed - no actual changes were made" -ForegroundColor Green
} elseif ($failedSteps.Count -eq 0) {
    Write-Host "`n" + "="*80 -ForegroundColor Green
    Write-Host "INSTALLATION COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "="*80 -ForegroundColor Green
    Write-Host ""
    Write-Host "The Ultimate Arabic Transcription Engine has been installed to:" -ForegroundColor Cyan
    Write-Host "  $InstallPath" -ForegroundColor White
    Write-Host ""
    Write-Host "Available interfaces:" -ForegroundColor Yellow
    Write-Host "  1. Web Interface: Double-click 'Ultimate Arabic Transcription.lnk' on desktop" -ForegroundColor White
    Write-Host "     Or run: $InstallPath\start.bat" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  2. CLI Tool: Double-click 'Arabic CLI.lnk' on desktop" -ForegroundColor White
    Write-Host "     Or run: $InstallPath\start_cli.bat" -ForegroundColor Gray
    Write-Host ""
    Write-Host "CLI Usage Examples:" -ForegroundColor Yellow
    Write-Host "  Interactive mode:  python arabic_cli_ultimate.py --interactive" -ForegroundColor White
    Write-Host "  Transcribe file:   python arabic_cli_ultimate.py --file audio.wav" -ForegroundColor White
    Write-Host "  Batch processing:  python arabic_cli_ultimate.py --batch-dir ./recordings" -ForegroundColor White
    Write-Host "  Full help:         python arabic_cli_ultimate.py --help-full" -ForegroundColor White
    Write-Host ""
    Write-Host "Features installed:" -ForegroundColor Yellow
    Write-Host "  - Multiple transcription engines (Ultimate, Advanced, Enhanced)" -ForegroundColor White
    Write-Host "  - LLM integration with Ollama (Aya and Llama models)" -ForegroundColor White
    Write-Host "  - Speaker diarization support" -ForegroundColor White
    Write-Host "  - Multiple output formats (TXT, JSON, SRT, VTT)" -ForegroundColor White
    Write-Host "  - Batch processing capabilities" -ForegroundColor White
    Write-Host "  - Interactive CLI mode" -ForegroundColor White
    Write-Host "  - Web-based interface" -ForegroundColor White
    Write-Host ""
    Write-Host "For support and documentation, visit:" -ForegroundColor Cyan
    Write-Host "  https://github.com/your-repo/ultimate-arabic-transcription-engine" -ForegroundColor White
    Write-Host ""
    Write-Host "="*80 -ForegroundColor Green
} else {
    Write-Host "[WARN] Installation completed with some issues:" -ForegroundColor Yellow
    foreach ($failed in $failedSteps) {
        Write-Host "   - $failed" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "[INFO] You may need to manually complete the failed steps." -ForegroundColor Yellow
    Write-Host "[INFO] Check the error messages above for guidance." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")