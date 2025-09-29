@echo off
REM Ultimate Arabic Transcription Engine - Simple Windows Installation
REM This is a simplified batch version of the PowerShell installer

echo 🎙️ Ultimate Arabic Transcription Engine - Windows Installation
echo ============================================================
echo.

REM Check for administrator privileges
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ❌ This script requires administrator privileges
    echo 💡 Please run as Administrator and try again
    pause
    exit /b 1
)

echo 📦 Installing Chocolatey package manager...
powershell -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"

REM Refresh environment variables
call refreshenv

echo 🐍 Installing Python 3.11...
choco install python311 -y --force

echo 🎬 Installing FFmpeg...
choco install ffmpeg -y --force

echo 📚 Installing Git...
choco install git -y --force

echo 🔧 Installing Visual C++ Redistributables...
choco install vcredist-all -y --force

REM Refresh environment variables again
call refreshenv

echo 📁 Creating installation directory...
if not exist "C:\ArabicSTT" mkdir "C:\ArabicSTT"
cd /d "C:\ArabicSTT"

echo 📦 Copying project files...
xcopy "%~dp0*" "C:\ArabicSTT\" /E /I /Y

echo 🌐 Creating Python virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo 📦 Upgrading pip...
python -m pip install --upgrade pip

echo 🔥 Installing PyTorch...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo 📚 Installing Python dependencies...
pip install -r requirements.txt

echo ⚙️ Creating environment configuration...
if not exist ".env" (
    echo # Ultimate Arabic Transcription Engine Configuration > .env
    echo PROCESSING_MODE=local >> .env
    echo WHISPER_MODEL_SIZE=medium >> .env
    echo WHISPER_DEVICE=auto >> .env
    echo HOST=0.0.0.0 >> .env
    echo PORT=5002 >> .env
    echo DEBUG=False >> .env
    echo LOG_LEVEL=INFO >> .env
)

echo 📁 Creating required directories...
if not exist "temp" mkdir "temp"
if not exist "uploads" mkdir "uploads"
if not exist "outputs" mkdir "outputs"
if not exist "models" mkdir "models"
if not exist "logs" mkdir "logs"

echo 🔧 Creating startup script...
echo @echo off > start.bat
echo echo Starting Ultimate Arabic Transcription Engine... >> start.bat
echo cd /d "C:\ArabicSTT" >> start.bat
echo call venv\Scripts\activate.bat >> start.bat
echo start /min python app.py >> start.bat
echo echo Service started. Access at http://localhost:5002 >> start.bat
echo timeout /t 3 /nobreak ^>nul >> start.bat

echo 🖥️ Creating desktop shortcut...
powershell -Command "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\Arabic Transcription Engine.lnk'); $Shortcut.TargetPath = 'C:\ArabicSTT\start.bat'; $Shortcut.WorkingDirectory = 'C:\ArabicSTT'; $Shortcut.Description = 'Ultimate Arabic Transcription Engine'; $Shortcut.Save()"

echo.
echo 🎉 Installation completed successfully!
echo.
echo 🚀 Quick Start:
echo 1. Double-click 'Arabic Transcription Engine' on your desktop
echo 2. Or run: C:\ArabicSTT\start.bat
echo 3. Open browser to: http://localhost:5002
echo.
echo Installation path: C:\ArabicSTT
echo.
pause