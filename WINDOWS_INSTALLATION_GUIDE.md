# 🪟 Windows 11 Installation Guide
## Ultimate Arabic Transcription Engine

This guide provides multiple installation methods for Windows 11, including fully automated silent installation options.

## 🚀 Quick Installation (Recommended)

### Method 1: PowerShell Script (Advanced)
**Fully automated, no user interaction required**

1. **Download the project** to your desired location
2. **Right-click on PowerShell** and select "Run as Administrator"
3. **Navigate to the project directory**:
   ```powershell
   cd "path\to\ultimate-arabic-transcription-engine"
   ```
4. **Run the installation script**:
   ```powershell
   .\install_windows.ps1
   ```

**Optional Parameters:**
```powershell
# Custom installation path
.\install_windows.ps1 -InstallPath "D:\MyArabicSTT"

# Skip specific components
.\install_windows.ps1 -SkipPython -SkipFFmpeg -SkipGit
```

### Method 2: Batch Script (Simple)
**Simplified installation for basic users**

1. **Right-click on Command Prompt** and select "Run as Administrator"
2. **Navigate to the project directory**:
   ```cmd
   cd "path\to\ultimate-arabic-transcription-engine"
   ```
3. **Run the batch installer**:
   ```cmd
   install_windows_simple.bat
   ```

## 📋 What Gets Installed

### System Dependencies
- **Chocolatey** - Package manager for Windows
- **Python 3.11** - Latest stable Python version
- **FFmpeg** - Audio/video processing library
- **Git** - Version control system
- **Visual C++ Redistributables** - Required runtime libraries

### Python Environment
- **Virtual Environment** - Isolated Python environment
- **PyTorch** - Machine learning framework (CPU version)
- **Whisper Models** - OpenAI Whisper for transcription
- **Flask** - Web framework for the interface
- **All Project Dependencies** - From requirements.txt

### Project Setup
- **Environment Configuration** (.env file)
- **Required Directories** (temp, uploads, outputs, models, logs)
- **Startup Scripts** (start.bat, service scripts)
- **Desktop Shortcut** - Quick access to the application

## 🛠️ Manual Installation

If you prefer to install manually or encounter issues with the automated scripts:

### Step 1: Install System Dependencies

1. **Install Chocolatey** (Package Manager):
   ```powershell
   Set-ExecutionPolicy Bypass -Scope Process -Force
   [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
   iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
   ```

2. **Install Python 3.11**:
   ```cmd
   choco install python311 -y
   ```

3. **Install FFmpeg**:
   ```cmd
   choco install ffmpeg -y
   ```

4. **Install Git** (optional):
   ```cmd
   choco install git -y
   ```

### Step 2: Setup Project Environment

1. **Create project directory**:
   ```cmd
   mkdir C:\ArabicSTT
   cd C:\ArabicSTT
   ```

2. **Copy project files** to the directory

3. **Create virtual environment**:
   ```cmd
   python -m venv venv
   venv\Scripts\activate.bat
   ```

4. **Install dependencies**:
   ```cmd
   python -m pip install --upgrade pip
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

### Step 3: Configure Environment

1. **Create .env file**:
   ```env
   PROCESSING_MODE=local
   WHISPER_MODEL_SIZE=medium
   WHISPER_DEVICE=auto
   HOST=0.0.0.0
   PORT=5002
   DEBUG=False
   LOG_LEVEL=INFO
   ```

2. **Create required directories**:
   ```cmd
   mkdir temp uploads outputs models logs
   ```

## 🚀 Starting the Application

### Option 1: Desktop Shortcut
- Double-click the "Arabic Transcription Engine" shortcut on your desktop

### Option 2: Batch Script
```cmd
cd C:\ArabicSTT
start.bat
```

### Option 3: Manual Start
```cmd
cd C:\ArabicSTT
venv\Scripts\activate.bat
python app.py
```

## 🌐 Accessing the Application

Once started, open your web browser and navigate to:
- **Local Access**: http://localhost:5002
- **Network Access**: http://YOUR_IP_ADDRESS:5002

## 🔧 Configuration Options

### Environment Variables (.env file)
```env
# Processing Mode
PROCESSING_MODE=local          # Use local Whisper models
# PROCESSING_MODE=api          # Use OpenAI API (requires API key)

# OpenAI API (if using API mode)
OPENAI_API_KEY=your_api_key_here

# Whisper Settings
WHISPER_MODEL_SIZE=medium      # small, medium, large, large-v2
WHISPER_DEVICE=auto           # auto, cpu, cuda (if GPU available)

# Server Settings
HOST=0.0.0.0                  # Listen on all interfaces
PORT=5002                     # Web interface port
DEBUG=False                   # Enable debug mode

# Security
SECRET_KEY=auto_generated     # Automatically generated
API_KEY_REQUIRED=False        # Require API key for access

# Logging
LOG_LEVEL=INFO               # DEBUG, INFO, WARNING, ERROR
```

### Web Interface Settings
Access the settings page at: http://localhost:5002/settings

## 🧪 Testing the Installation

### Quick Test
1. Start the application
2. Open http://localhost:5002
3. Upload a short Arabic audio file
4. Verify transcription works

### CLI Test
```cmd
cd C:\ArabicSTT
venv\Scripts\activate.bat
python arabic_cli_ultimate.py --file "path\to\audio.mp3"
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "Python not found"
- Ensure Python is installed and in PATH
- Restart Command Prompt/PowerShell after installation
- Try: `refreshenv` (if using Chocolatey)

#### 2. "FFmpeg not found"
- Install FFmpeg: `choco install ffmpeg -y`
- Restart terminal and try again

#### 3. "Permission denied"
- Run Command Prompt/PowerShell as Administrator
- Check Windows Defender/Antivirus settings

#### 4. "Module not found" errors
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

#### 5. "Port already in use"
- Change port in .env file: `PORT=5003`
- Or stop other services using port 5002

### Performance Optimization

#### For CPU-only systems:
```env
WHISPER_DEVICE=cpu
WHISPER_MODEL_SIZE=small    # Faster processing
```

#### For systems with NVIDIA GPU:
1. Install CUDA toolkit
2. Install GPU version of PyTorch:
   ```cmd
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Set in .env:
   ```env
   WHISPER_DEVICE=cuda
   ```

## 📁 Directory Structure

After installation, your directory structure will be:
```
C:\ArabicSTT\
├── venv\                    # Python virtual environment
├── core\                    # Core transcription engines
├── templates\               # Web interface templates
├── static\                  # Web assets
├── temp\                    # Temporary files
├── uploads\                 # Uploaded audio files
├── outputs\                 # Transcription outputs
├── models\                  # Whisper models cache
├── logs\                    # Application logs
├── .env                     # Environment configuration
├── requirements.txt         # Python dependencies
├── app.py                   # Main web application
├── start.bat               # Startup script
└── README.md               # Documentation
```

## 🔄 Updating the System

To update to a newer version:

1. **Backup your configuration**:
   ```cmd
   copy .env .env.backup
   ```

2. **Download new version** and replace files

3. **Update dependencies**:
   ```cmd
   venv\Scripts\activate.bat
   pip install -r requirements.txt --upgrade
   ```

4. **Restore configuration**:
   ```cmd
   copy .env.backup .env
   ```

## 🆘 Support

If you encounter issues:

1. **Check the logs**: `logs\app.log`
2. **Verify installation**: Run the test commands above
3. **Check system requirements**: Windows 11, 8GB+ RAM recommended
4. **Review error messages**: Most issues are dependency-related

## 📊 System Requirements

### Minimum Requirements
- **OS**: Windows 11 (Windows 10 may work)
- **RAM**: 4GB (8GB recommended)
- **Storage**: 5GB free space
- **Internet**: Required for initial setup and model downloads

### Recommended Requirements
- **OS**: Windows 11
- **RAM**: 8GB or more
- **Storage**: 10GB+ free space
- **CPU**: Multi-core processor
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster processing)

## 🔐 Security Notes

- The installation scripts require administrator privileges
- All dependencies are installed from official sources
- No sensitive data is collected or transmitted
- Local processing ensures privacy of audio files
- Web interface runs locally by default (localhost:5002)

---

**Need help?** Check the main README.md for additional documentation and examples.