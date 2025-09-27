#!/usr/bin/env python3
"""
Setup script for Arabic STT Platform
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("  Arabic Speech-to-Text Platform Setup")
    print("  Self-hosted STT with Arabic dialect support")
    print("=" * 60)
    print()

def check_python_version():
    """Check Python version compatibility."""
    if sys.version_info < (3, 9):
        print("❌ Error: Python 3.9 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version}")

def check_ffmpeg():
    """Check if FFmpeg is installed."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ FFmpeg is installed")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ FFmpeg not found")
    print("Please install FFmpeg:")
    
    system = platform.system()
    if system == "Darwin":  # macOS
        print("  brew install ffmpeg")
    elif system == "Linux":
        print("  # Ubuntu/Debian:")
        print("  sudo apt update && sudo apt install ffmpeg")
        print("  # CentOS/RHEL:")
        print("  sudo yum install ffmpeg")
    elif system == "Windows":
        print("  Download from: https://ffmpeg.org/download.html")
    
    return False

def install_requirements():
    """Install Python requirements."""
    print("\n📦 Installing Python requirements...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        'uploads',
        'results', 
        'temp',
        'models',
        'logs'
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")

def download_models():
    """Download initial Whisper models."""
    print("\n🤖 Downloading Whisper models...")
    
    try:
        from faster_whisper import WhisperModel
        
        # Download medium model (recommended)
        print("  Downloading medium model...")
        model = WhisperModel("medium", download_root="./models")
        print("  ✅ Medium model downloaded")
        
        # Optionally download small model for faster processing
        print("  Downloading small model...")
        model = WhisperModel("small", download_root="./models")
        print("  ✅ Small model downloaded")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model download failed: {e}")
        print("  Models will be downloaded on first use.")
        return False

def setup_rnnoise():
    """Setup RNNoise if available."""
    print("\n🔊 Checking RNNoise availability...")
    
    try:
        result = subprocess.run(['which', 'rnnoise_demo'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ RNNoise is available")
            return True
    except Exception:
        pass
    
    print("  ⚠️  RNNoise not found (optional)")
    print("  For advanced noise reduction, install RNNoise:")
    print("  https://github.com/xiph/rnnoise")
    return False

def create_config_file():
    """Create default configuration file."""
    print("\n⚙️  Creating configuration...")
    
    config_content = '''# Arabic STT Platform Configuration
# Copy this to .env for production use

# Server settings
HOST=0.0.0.0
PORT=5000
DEBUG=False

# Security
SECRET_KEY=change-this-in-production
API_KEY_REQUIRED=False
API_KEY=arabic-stt-api-key

# Processing settings
WHISPER_MODEL_SIZE=medium
WHISPER_DEVICE=auto
ENABLE_RNNOISE=True

# Logging
LOG_LEVEL=INFO

# Redis (for production)
REDIS_URL=redis://localhost:6379/0
'''
    
    with open('.env.example', 'w') as f:
        f.write(config_content)
    
    print("  ✅ Configuration template created (.env.example)")

def run_tests():
    """Run basic tests."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        import flask
        import faster_whisper
        import librosa
        print("  ✅ Core imports successful")
        
        # Test configuration
        from config import Config
        config = Config()
        print("  ✅ Configuration loaded")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Tests failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Copy .env.example to .env and configure as needed")
    print("2. Start the server:")
    print("   python app.py")
    print()
    print("3. Open your browser to:")
    print("   http://localhost:5000")
    print()
    print("4. For production deployment:")
    print("   - Set DEBUG=False in .env")
    print("   - Configure proper SECRET_KEY") 
    print("   - Enable API authentication")
    print("   - Setup reverse proxy (nginx)")
    print()
    print("For Arabic dialect optimization:")
    print("- Consider fine-tuning models with Iraqi Arabic data")
    print("- Install RNNoise for better noise reduction")
    print("- Use GPU acceleration for faster processing")
    print()

def main():
    """Main setup function."""
    print_banner()
    
    # Check prerequisites
    check_python_version()
    ffmpeg_ok = check_ffmpeg()
    
    if not ffmpeg_ok:
        print("\n❌ Setup cannot continue without FFmpeg")
        sys.exit(1)
    
    # Install and configure
    if not install_requirements():
        sys.exit(1)
    
    create_directories()
    download_models()
    setup_rnnoise()
    create_config_file()
    
    # Test setup
    if not run_tests():
        print("\n⚠️  Setup completed with warnings")
    
    print_next_steps()

if __name__ == '__main__':
    main()
